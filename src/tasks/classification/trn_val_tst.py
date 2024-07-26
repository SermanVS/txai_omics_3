from typing import List, Optional
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers.logger import Logger
import xgboost as xgb
from catboost import CatBoost, Pool
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from src.datamodules.cross_validation import RepeatedStratifiedKFoldCVSplitter
from src.datamodules.tabular import TabularDataModule
import numpy as np
from src.utils import utils
import pandas as pd
from tqdm import tqdm
from src.tasks.classification.shap import explain_shap, get_feature_importance
from src.tasks.routines import eval_classification, eval_loss, save_feature_importance
from datetime import datetime
from pathlib import Path
import pickle
import wandb
import glob
import os
from src.models.tabular.base import get_model_framework_dict


log = utils.get_logger(__name__)


def trn_val_tst_classification(config: DictConfig) -> Optional[float]:

    if "seed" in config:
        seed_everything(config.seed, workers=True)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    model_framework_dict = get_model_framework_dict()
    model_framework = model_framework_dict[config.model.name]

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.perform_split()
    features = datamodule.get_features()
    num_features = len(features['all'])
    config.in_dim = num_features
    class_names = datamodule.get_class_names()
    target = datamodule.target
    target_label = datamodule.target_label
    if datamodule.target_classes_num != config.out_dim:
        raise ValueError(f"Inconsistent out_dim. From datamodule: {datamodule.target_classes_num}, from config: {config.out_dim}")
    df = datamodule.get_data()
    df["pred"] = 0
    ids_tst = datamodule.ids_tst

    cv_splitter = RepeatedStratifiedKFoldCVSplitter(
        datamodule=datamodule,
        is_split=config.cv_is_split,
        n_splits=config.cv_n_splits,
        n_repeats=config.cv_n_repeats,
        random_state=config.seed,
        all_combinations=config.cv_all_combinations
    )

    best = {}
    if config.direction == "min":
        best["optimized_metric"] = np.Inf
    elif config.direction == "max":
        best["optimized_metric"] = 0.0

    metrics_cv = pd.DataFrame(columns=['fold', 'optimized_metric'])
    feature_importances_cv = pd.DataFrame(columns=['fold'] + features['all'])

    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = config.callbacks.model_checkpoint.filename

    for fold_idx, (ids_trn, ids_val) in tqdm(enumerate(cv_splitter.split())):
        datamodule.ids_trn = ids_trn
        datamodule.ids_val = ids_val
        datamodule.refresh_datasets()
        X_trn = df.loc[df.index[ids_trn], features['all']]
        y_trn = df.loc[df.index[ids_trn], target]
        df.loc[df.index[ids_trn], f"fold_{fold_idx:04d}"] = "trn"
        X_val = df.loc[df.index[ids_val], features['all']]
        y_val = df.loc[df.index[ids_val], target]
        df.loc[df.index[ids_val], f"fold_{fold_idx:04d}"] = "val"
        X_tst = {}
        y_tst = {}
        for tst_set_name in ids_tst:
            X_tst[tst_set_name] = df.loc[df.index[ids_tst[tst_set_name]], features['all']]
            y_tst[tst_set_name] = df.loc[df.index[ids_tst[tst_set_name]], target]
            if tst_set_name != 'tst_all':
                df.loc[df.index[ids_tst[tst_set_name]], f"fold_{fold_idx:04d}"] = tst_set_name

        ckpt_curr = ckpt_name + f"_fold_{fold_idx:04d}"
        if 'csv' in config.logger:
            config.logger.csv["version"] = f"fold_{fold_idx}"
        if 'wandb' in config.logger:
            config.logger.wandb["version"] = f"fold_{fold_idx}_{start_time}"

        if model_framework == "pytorch":
            # Init lightning model
            widedeep = datamodule.get_widedeep()
            embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
            categorical_cardinality = [x[1] for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
            if config.model.name.startswith('widedeep'):
                config.model.column_idx = widedeep['column_idx']
                config.model.cat_embed_input = widedeep['cat_embed_input']
                config.model.continuous_cols = widedeep['continuous_cols']
            elif config.model.name.startswith('pytorch_tabular'):
                config.model.continuous_cols = features['con']
                config.model.categorical_cols = features['cat']
                config.model.embedding_dims = embedding_dims
                config.model.categorical_cardinality = categorical_cardinality
            elif config.model.name == 'nam':
                num_unique_vals = [len(np.unique(X_trn.loc[:, f].values)) for f in features['all']]
                num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in num_unique_vals]
                config.model.num_units = num_units

            log.info(f"Instantiating model <{config.model._target_}>")
            model = hydra.utils.instantiate(config.model)
            if config.print_model:
                print(model)

            # Init lightning callbacks
            config.callbacks.model_checkpoint.filename = ckpt_curr
            callbacks: List[Callback] = []
            if "callbacks" in config:
                for _, cb_conf in config.callbacks.items():
                    if "_target_" in cb_conf:
                        log.info(f"Instantiating callback <{cb_conf._target_}>")
                        callbacks.append(hydra.utils.instantiate(cb_conf))

        # Init lightning loggers
        loggers: List[Logger] = []
        if "logger" in config:
            for _, lg_conf in config.logger.items():
                if "_target_" in lg_conf:
                    log.info(f"Instantiating logger <{lg_conf._target_}>")
                    loggers.append(hydra.utils.instantiate(lg_conf))

        if model_framework == "pytorch":
            # Init lightning trainer
            log.info(f"Instantiating trainer <{config.trainer._target_}>")
            trainer: Trainer = hydra.utils.instantiate(
                config.trainer, callbacks=callbacks, logger=loggers, _convert_="partial"
            )
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters_pytorch(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )
        elif model_framework == "stand_alone":
            log.info("Logging hyperparameters!")
            utils.log_hyperparameters_stand_alone(
                config=config,
                logger=loggers,
            )
        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        # Train the model
        if model_framework == "pytorch":
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule)

            # Evaluate model on test set, using the best model achieved during training
            if config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
                log.info("Starting testing!")
                tst_dataloaders = datamodule.test_dataloaders()
                if len(tst_dataloaders) > 0:
                    if 'tst_all' in tst_dataloaders:
                        tst_dataloader = tst_dataloaders['tst_all']
                    else:
                        tst_dataloader = tst_dataloaders[list(tst_dataloaders.keys())[0]]
                    if tst_dataloader is not None and len(tst_dataloader) > 0:
                        trainer.test(model, tst_dataloader, ckpt_path="best")
                    else:
                        log.info("Test data is empty!")

            datamodule.dataloaders_evaluate = True
            trn_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()
            tst_dataloaders = datamodule.test_dataloaders()
            datamodule.dataloaders_evaluate = False

            model.produce_probabilities = True
            y_trn_pred_prob = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            y_val_pred_prob = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            y_tst_pred_prob = {}
            for tst_set_name in ids_tst:
                y_tst_pred_prob[tst_set_name] = torch.cat(trainer.predict(model, dataloaders=tst_dataloaders[tst_set_name], return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            model.produce_probabilities = False
            y_trn_pred_raw = torch.cat(trainer.predict(model, dataloaders=trn_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            y_val_pred_raw = torch.cat(trainer.predict(model, dataloaders=val_dataloader, return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
            y_tst_pred_raw = {}
            y_trn_pred = np.argmax(y_trn_pred_prob, 1)
            y_val_pred = np.argmax(y_val_pred_prob, 1)
            y_tst_pred = {}
            for tst_set_name in ids_tst:
                y_tst_pred_raw[tst_set_name] = torch.cat(trainer.predict(model, dataloaders=tst_dataloaders[tst_set_name], return_predictions=True, ckpt_path="best")).cpu().detach().numpy()
                y_tst_pred[tst_set_name] = np.argmax(y_tst_pred_prob[tst_set_name], 1)
            model.produce_probabilities = True

            # Feature importance
            if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                model = type(model).load_from_checkpoint(
                    checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                model.eval()
                model.freeze()
            feature_importances = model.get_feature_importance(X_trn.values.astype('float32'), features, config.feature_importance)

        elif model_framework == "stand_alone":
            if config.model.name == "xgboost":
                model_params = {
                    'num_class': config.model.output_dim,
                    'booster': config.model.booster,
                    'eta': config.model.learning_rate,
                    'max_depth': config.model.max_depth,
                    'gamma': config.model.gamma,
                    'sampling_method': config.model.sampling_method,
                    'subsample': config.model.subsample,
                    'objective': config.model.objective,
                    'verbosity': config.model.verbosity,
                    'eval_metric': config.model.eval_metric,
                }

                dmat_trn = xgb.DMatrix(X_trn, y_trn, feature_names=features['all'], enable_categorical=True)
                dmat_val = xgb.DMatrix(X_val, y_val, feature_names=features['all'], enable_categorical=True)
                dmat_tst = {}
                for tst_set_name in ids_tst:
                    dmat_tst[tst_set_name] = xgb.DMatrix(X_tst[tst_set_name], y_tst[tst_set_name], feature_names=features['all'], enable_categorical=True)

                evals_result = {}
                model = xgb.train(
                    params=model_params,
                    dtrain=dmat_trn,
                    evals=[(dmat_trn, "train"), (dmat_val, "val")],
                    num_boost_round=config.max_epochs,
                    early_stopping_rounds=config.patience,
                    evals_result=evals_result,
                    verbose_eval=False
                )

                y_trn_pred_prob = model.predict(dmat_trn)
                y_val_pred_prob = model.predict(dmat_val)
                y_tst_pred_prob = {}
                y_trn_pred_raw = model.predict(dmat_trn, output_margin=True)
                y_val_pred_raw = model.predict(dmat_val, output_margin=True)
                y_tst_pred_raw = {}
                y_trn_pred = np.argmax(y_trn_pred_prob, 1)
                y_val_pred = np.argmax(y_val_pred_prob, 1)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred_prob[tst_set_name] = model.predict(dmat_tst[tst_set_name])
                    y_tst_pred_raw[tst_set_name] = model.predict(dmat_tst[tst_set_name], output_margin=True)
                    y_tst_pred[tst_set_name] = np.argmax(y_tst_pred_prob[tst_set_name], 1)

                loss_info = {
                    'epoch': list(range(len(evals_result['train'][config.model.eval_metric]))),
                    'trn/loss': evals_result['train'][config.model.eval_metric],
                    'val/loss': evals_result['val'][config.model.eval_metric]
                }

                if config.feature_importance.startswith("shap"):
                    def predict_func(X):
                        X = xgb.DMatrix(X, feature_names=features['all'], enable_categorical=True)
                        y = model.predict(X)
                        return y

                    fi_data = {
                        'feature_importance': config.feature_importance,
                        'model': model,
                        'X': X_trn,
                        'y_pred_prob': y_trn_pred_prob,
                        'y_pred_raw': y_trn_pred_raw,
                        'predict_func': predict_func,
                        'features': features
                    }
                    feature_importances = get_feature_importance(fi_data)
                else:
                    if config.feature_importance == "native":
                        fi = model.get_score(importance_type='weight')
                        fi_features = list(fi.keys())
                        fi_importances = list(fi.values())
                    elif config.feature_importance == "none":
                        fi_features = features['all']
                        fi_importances = np.zeros(len(features['all']))
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    feature_importances = pd.DataFrame.from_dict(
                        {
                            'feature': fi_features,
                            'importance': fi_importances
                        }
                    )

            elif config.model.name == "catboost":
                model_params = {
                    'classes_count': config.model.output_dim,
                    'loss_function': config.model.loss_function,
                    'learning_rate': config.model.learning_rate,
                    'depth': config.model.depth,
                    'min_data_in_leaf': config.model.min_data_in_leaf,
                    'max_leaves': config.model.max_leaves,
                    'task_type': config.model.task_type,
                    'verbose': config.model.verbose,
                    'iterations': config.model.max_epochs,
                    'early_stopping_rounds': config.model.patience
                }

                trn_pool = Pool(X_trn, label=y_trn, feature_names=features['all'], cat_features=features['cat'])
                val_pool = Pool(X_val, label=y_val, feature_names=features['all'], cat_features=features['cat'])

                model = CatBoost(params=model_params)
                model.fit(trn_pool, eval_set=val_pool, use_best_model=True)
                model.set_feature_names(features['all'])

                y_trn_pred_prob = model.predict(X_trn, prediction_type="Probability")
                y_val_pred_prob = model.predict(X_val, prediction_type="Probability")
                y_tst_pred_prob = {}
                y_trn_pred_raw = model.predict(X_trn, prediction_type="RawFormulaVal")
                y_val_pred_raw = model.predict(X_val, prediction_type="RawFormulaVal")
                y_tst_pred_raw = {}
                y_trn_pred = np.argmax(y_trn_pred_prob, 1)
                y_val_pred = np.argmax(y_val_pred_prob, 1)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred_prob[tst_set_name] = model.predict(X_tst[tst_set_name], prediction_type="Probability")
                    y_tst_pred_raw[tst_set_name] = model.predict(X_tst[tst_set_name], prediction_type="RawFormulaVal")
                    y_tst_pred[tst_set_name] = np.argmax(y_tst_pred_prob[tst_set_name], 1)

                metrics_trn = pd.read_csv(f"catboost_info/learn_error.tsv", delimiter="\t")
                metrics_val = pd.read_csv(f"catboost_info/test_error.tsv", delimiter="\t")
                loss_info = {
                    'epoch': metrics_trn.iloc[:, 0],
                    'trn/loss': metrics_trn.iloc[:, 1],
                    'val/loss': metrics_val.iloc[:, 1]
                }

                if config.feature_importance.startswith("shap"):
                    def predict_func(X):
                        X = pd.DataFrame(data=X, columns=features["all"])
                        X[features["cat"]] = X[features["cat"]].astype('int32')
                        y = model.predict(X)
                        return y

                    fi_data = {
                        'feature_importance': config.feature_importance,
                        'model': model,
                        'X': X_trn,
                        'y_pred_prob': y_trn_pred_prob,
                        'y_pred_raw': y_trn_pred_raw,
                        'predict_func': predict_func,
                        'features': features
                    }
                    feature_importances = get_feature_importance(fi_data)
                else:
                    if config.feature_importance == "native":
                        fi_features = model.feature_names_
                        fi_importances = list(model.feature_importances_)
                    elif config.feature_importance == "none":
                        fi_features = features['all']
                        fi_importances = np.zeros(len(features['all']))
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    feature_importances = pd.DataFrame.from_dict(
                        {
                            'feature': fi_features,
                            'importance': fi_importances
                        }
                    )

            elif config.model.name == "lightgbm":
                model_params = {
                    'num_class': config.model.output_dim,
                    'objective': config.model.objective,
                    'boosting': config.model.boosting,
                    'learning_rate': config.model.learning_rate,
                    'num_leaves': config.model.num_leaves,
                    'device': config.model.device,
                    'max_depth': config.model.max_depth,
                    'min_data_in_leaf': config.model.min_data_in_leaf,
                    'feature_fraction': config.model.feature_fraction,
                    'bagging_fraction': config.model.bagging_fraction,
                    'bagging_freq': config.model.bagging_freq,
                    'verbose': config.model.verbose,
                    'metric': config.model.metric
                }

                ds_trn = lightgbm.Dataset(X_trn, label=y_trn, feature_name=features['all'], categorical_feature=features['cat'])
                ds_val = lightgbm.Dataset(X_val, label=y_val, reference=ds_trn, feature_name=features['all'], categorical_feature=features['cat'])

                evals_result = {}
                model = lightgbm.train(
                    params=model_params,
                    train_set=ds_trn,
                    num_boost_round=config.max_epochs,
                    valid_sets=[ds_val, ds_trn],
                    valid_names=['val', 'train'],
                    evals_result=evals_result,
                    early_stopping_rounds=config.patience,
                    verbose_eval=False
                )

                y_trn_pred_prob = model.predict(X_trn, num_iteration=model.best_iteration)
                y_val_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
                y_tst_pred_prob = {}
                y_trn_pred_raw = model.predict(X_trn, num_iteration=model.best_iteration, raw_score=True)
                y_val_pred_raw = model.predict(X_val, num_iteration=model.best_iteration, raw_score=True)
                y_tst_pred_raw = {}
                y_trn_pred = np.argmax(y_trn_pred_prob, 1)
                y_val_pred = np.argmax(y_val_pred_prob, 1)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred_prob[tst_set_name] = model.predict(X_tst[tst_set_name], num_iteration=model.best_iteration)
                    y_tst_pred_raw[tst_set_name] = model.predict(X_tst[tst_set_name], num_iteration=model.best_iteration, raw_score=True)
                    y_tst_pred[tst_set_name] = np.argmax(y_tst_pred_prob[tst_set_name], 1)

                loss_info = {
                    'epoch': list(range(len(evals_result['train'][config.model.metric]))),
                    'trn/loss': evals_result['train'][config.model.metric],
                    'val/loss': evals_result['val'][config.model.metric]
                }

                if config.feature_importance.startswith("shap"):
                    def predict_func(X):
                        y = model.predict(X, num_iteration=model.best_iteration)
                        return y

                    fi_data = {
                        'feature_importance': config.feature_importance,
                        'model': model,
                        'X': X_trn,
                        'y_pred_prob': y_trn_pred_prob,
                        'y_pred_raw': y_trn_pred_raw,
                        'predict_func': predict_func,
                        'features': features
                    }
                    feature_importances = get_feature_importance(fi_data)
                else:
                    if config.feature_importance == "native":
                        fi_features = model.feature_name()
                        fi_importances = list(model.feature_importance())
                    elif config.feature_importance == "none":
                        fi_features = features['all']
                        fi_importances = np.zeros(len(features['all']))
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    feature_importances = pd.DataFrame.from_dict(
                        {
                            'feature': fi_features,
                            'importance': fi_importances
                        }
                    )

            elif config.model.name == "logistic_regression":
                model = LogisticRegression(
                    penalty=config.model.penalty,
                    l1_ratio=config.model.l1_ratio,
                    C=config.model.C,
                    multi_class=config.model.multi_class,
                    solver=config.model.solver,
                    max_iter=config.model.max_iter,
                    tol=config.model.tol,
                    verbose=config.model.verbose,
                    class_weight='balanced'
                ).fit(X_trn, y_trn)

                y_trn_pred_prob = model.predict_proba(X_trn)
                y_val_pred_prob = model.predict_proba(X_val)
                y_tst_pred_prob = {}
                y_trn_pred_raw = model.predict_proba(X_trn)
                y_val_pred_raw = model.predict_proba(X_val)
                y_tst_pred_raw = {}
                y_trn_pred = model.predict(X_trn)
                y_val_pred = model.predict(X_val)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred_prob[tst_set_name] = model.predict_proba(X_tst[tst_set_name])
                    y_tst_pred_raw[tst_set_name] = model.predict_proba(X_tst[tst_set_name])
                    y_tst_pred[tst_set_name] = model.predict(X_tst[tst_set_name])

                loss_info = {
                    'epoch': [0],
                    'trn/loss': [0],
                    'val/loss': [0]
                }

                if config.feature_importance.startswith("shap"):
                    def predict_func(X):
                        y = model.predict_proba(X)
                        return y

                    fi_data = {
                        'feature_importance': config.feature_importance,
                        'model': model,
                        'X': X_trn,
                        'y_pred_prob': y_trn_pred_prob,
                        'y_pred_raw': y_trn_pred_raw,
                        'predict_func': predict_func,
                        'features': features
                    }
                    feature_importances = get_feature_importance(fi_data)
                else:
                    if config.feature_importance == "native":
                        fi_features = ['Intercept'] + features['all']
                        fi_importances = [model.intercept_] + list(model.coef_.ravel())
                    elif config.feature_importance == "none":
                        fi_features = features['all']
                        fi_importances = np.zeros(len(features['all']))
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    feature_importances = pd.DataFrame.from_dict(
                        {
                            'feature': fi_features,
                            'importance': fi_importances
                        }
                    )

            elif config.model.name == "svm":
                model = svm.SVC(
                    C=config.model.C,
                    kernel=config.model.kernel,
                    decision_function_shape=config.model.decision_function_shape,
                    max_iter=config.model.max_iter,
                    tol=config.model.tol,
                    verbose=config.model.verbose,
                    probability=True,
                    class_weight='balanced'
                ).fit(X_trn, y_trn)

                y_trn_pred_prob = model.predict_proba(X_trn)
                y_val_pred_prob = model.predict_proba(X_val)
                y_tst_pred_prob = {}
                y_trn_pred_raw = model.predict_proba(X_trn)
                y_val_pred_raw = model.predict_proba(X_val)
                y_tst_pred_raw = {}
                y_trn_pred = model.predict(X_trn)
                y_val_pred = model.predict(X_val)
                y_tst_pred = {}
                for tst_set_name in ids_tst:
                    y_tst_pred_prob[tst_set_name] = model.predict_proba(X_tst[tst_set_name])
                    y_tst_pred_raw[tst_set_name] = model.predict_proba(X_tst[tst_set_name])
                    y_tst_pred[tst_set_name] = model.predict(X_tst[tst_set_name])

                loss_info = {
                    'epoch': [0],
                    'trn/loss': [0],
                    'val/loss': [0]
                }

                if config.feature_importance.startswith("shap"):
                    def predict_func(X):
                        y = model.predict_proba(X)
                        return y

                    fi_data = {
                        'feature_importance': config.feature_importance,
                        'model': model,
                        'X': X_trn,
                        'y_pred_prob': y_trn_pred_prob,
                        'y_pred_raw': y_trn_pred_raw,
                        'predict_func': predict_func,
                        'features': features
                    }
                    feature_importances = get_feature_importance(fi_data)
                else:
                    if config.feature_importance == "none":
                        fi_features = features['all']
                        fi_importances = np.zeros(len(features['all']))
                    else:
                        raise ValueError(f"Unsupported feature importance method: {config.feature_importance}")
                    feature_importances = pd.DataFrame.from_dict(
                        {
                            'feature': fi_features,
                            'importance': fi_importances
                        }
                    )

            else:
                raise ValueError(f"Model {config.model.name} is not supported")

        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        metrics_trn = eval_classification(config, class_names, y_trn.values, y_trn_pred, y_trn_pred_prob, loggers, 'trn', is_log=True, is_save=False)
        metrics_all = metrics_trn.copy()
        for m in metrics_trn.index.values:
            metrics_cv.at[fold_idx, f"trn_{m}"] = metrics_trn.at[m, 'trn']
        metrics_val = eval_classification(config, class_names, y_val.values, y_val_pred, y_val_pred_prob, loggers, 'val', is_log=True, is_save=False)
        metrics_all.loc[metrics_all.index.values, "val"] = metrics_val.loc[metrics_all.index.values, "val"]
        for m in metrics_val.index.values:
            metrics_cv.at[fold_idx, f"val_{m}"] = metrics_val.at[m, 'val']
        metrics_tst = {}
        for tst_set_name in ids_tst:
            metrics_tst[tst_set_name] = eval_classification(config, class_names, y_tst[tst_set_name].values, y_tst_pred[tst_set_name], y_tst_pred_prob[tst_set_name], loggers, tst_set_name, is_log=True, is_save=False)
            metrics_all.loc[metrics_all.index.values, tst_set_name] = metrics_tst[tst_set_name].loc[metrics_all.index.values, tst_set_name]
            for m in metrics_tst[tst_set_name].index.values:
                metrics_cv.at[fold_idx, f"{tst_set_name}_{m}"] = metrics_tst[tst_set_name].at[m, tst_set_name]
        metrics_all["trn_val"] = metrics_all.loc[:, ['trn', 'val']].sum(axis=1) / 2
        for tst_set_name in ids_tst:
            metrics_all[f"trn_val_{tst_set_name}"] = metrics_all.loc[:, ['trn', 'val', tst_set_name]].sum(axis=1) / 3
            metrics_all[f"val_{tst_set_name}"] = metrics_all.loc[:, ['val', tst_set_name]].sum(axis=1) / 2

        # Make sure everything closed properly
        if model_framework == "pytorch":
            log.info("Finalizing!")
            utils.finish(
                config=config,
                model=model,
                datamodule=datamodule,
                trainer=trainer,
                callbacks=callbacks,
                logger=loggers,
            )
        elif model_framework == "stand_alone":
            if 'wandb' in config.logger:
                wandb.define_metric(f"epoch")
                wandb.define_metric(f"trn/loss")
                wandb.define_metric(f"val/loss")
            eval_loss(loss_info, loggers, is_log=True, is_save=False)
            for logger in loggers:
                logger.save()
            if 'wandb' in config.logger:
                wandb.finish()
        else:
            raise ValueError(f"Unsupported model_framework: {model_framework}")

        if config.direction == "min":
            if metrics_all.at[config.optimized_metric, config.optimized_part] < best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False
        elif config.direction == "max":
            if metrics_all.at[config.optimized_metric, config.optimized_part] > best["optimized_metric"]:
                is_renew = True
            else:
                is_renew = False

        if is_renew:
            best["optimized_metric"] = metrics_all.at[config.optimized_metric, config.optimized_part]
            if model_framework == "pytorch":
                if Path(f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt").is_file():
                    model = type(model).load_from_checkpoint(checkpoint_path=f"{config.callbacks.model_checkpoint.dirpath}{config.callbacks.model_checkpoint.filename}.ckpt")
                    model.eval()
                    model.freeze()
                best["model"] = model

                def predict_func(X):
                    best["model"].produce_probabilities = True
                    batch = {
                        'all': torch.from_numpy(np.float32(X[:, features['all_ids']])),
                        'continuous': torch.from_numpy(np.float32(X[:, features['con_ids']])),
                        'categorical': torch.from_numpy(np.int32(X[:, features['cat_ids']])),
                    }
                    tmp = best["model"](batch)
                    return tmp.cpu().detach().numpy()

            elif model_framework == "stand_alone":
                best["model"] = model
                best['loss_info'] = loss_info

                if config.model.name == "xgboost":
                    def predict_func(X):
                        X = xgb.DMatrix(X, feature_names=features['all'], enable_categorical=True)
                        y = best["model"].predict(X)
                        return y
                elif config.model.name == "catboost":
                    def predict_func(X):
                        X = pd.DataFrame(data=X, columns=features["all"])
                        X[features["cat"]] = X[features["cat"]].astype('int32')
                        y = best["model"].predict(X)
                        return y
                elif config.model.name == "lightgbm":
                    def predict_func(X):
                        y = best["model"].predict(X, num_iteration=best["model"].best_iteration)
                        return y
                elif config.model.name == "logistic_regression":
                    def predict_func(X):
                        y = best["model"].predict_proba(X)
                        return y
                elif config.model.name == "svm":
                    def predict_func(X):
                        y = best["model"].predict_proba(X)
                        return y
                else:
                    raise ValueError(f"Model {config.model.name} is not supported")

            else:
                raise ValueError(f"Unsupported model_framework: {model_framework}")

            best['predict_func'] = predict_func
            best['feature_importances'] = feature_importances
            best['fold'] = fold_idx
            best['ids_trn'] = ids_trn
            best['ids_val'] = ids_val
            df.loc[df.index[ids_trn], "pred"] = y_trn_pred
            df.loc[df.index[ids_val], "pred"] = y_val_pred
            for cl_id, cl in enumerate(class_names):
                df.loc[df.index[ids_trn], f"pred_prob_{cl_id}"] = y_trn_pred_prob[:, cl_id]
                df.loc[df.index[ids_val], f"pred_prob_{cl_id}"] = y_val_pred_prob[:, cl_id]
                df.loc[df.index[ids_trn], f"pred_raw_{cl_id}"] = y_trn_pred_raw[:, cl_id]
                df.loc[df.index[ids_val], f"pred_raw_{cl_id}"] = y_val_pred_raw[:, cl_id]
            for tst_set_name in ids_tst:
                if tst_set_name != 'tst_all':
                    df.loc[df.index[ids_tst[tst_set_name]], "pred"] = y_tst_pred[tst_set_name]
                    for cl_id, cl in enumerate(class_names):
                        df.loc[df.index[ids_tst[tst_set_name]], f"pred_prob_{cl_id}"] = y_tst_pred_prob[tst_set_name][:, cl_id]
                        df.loc[df.index[ids_tst[tst_set_name]], f"pred_raw_{cl_id}"] = y_tst_pred_raw[tst_set_name][:, cl_id]

        if model_framework == "pytorch":
            fns = glob.glob(f"{ckpt_name}*.ckpt")
            fns.remove(f"{ckpt_name}_fold_{best['fold']:04d}.ckpt")
            for fn in fns:
                os.remove(fn)

        metrics_cv.at[fold_idx, 'fold'] = fold_idx
        metrics_cv.at[fold_idx, 'optimized_metric'] = metrics_all.at[config.optimized_metric, config.optimized_part]
        feature_importances_cv.at[fold_idx, 'fold'] = fold_idx
        for feat in features['all']:
            feature_importances_cv.at[fold_idx, feat] = feature_importances.loc[feature_importances['feature'] == feat, 'importance'].values[0]

    metrics_cv.to_excel(f"metrics_cv.xlsx", index=False)
    feature_importances_cv.to_excel(f"feature_importances_cv.xlsx", index=False)
    cv_ids_cols = [f"fold_{fold_idx:04d}" for fold_idx in metrics_cv.loc[:,'fold'].values]
    if datamodule.split_top_feat:
        cv_ids_cols.append(datamodule.split_top_feat)
    cv_ids = df.loc[:, cv_ids_cols]
    cv_ids.to_excel(f"cv_ids.xlsx", index=True)
    predictions_cols = [f"fold_{best['fold']:04d}", target, "pred"] + [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]
    if datamodule.split_top_feat:
        predictions_cols.append(datamodule.split_top_feat)
    predictions = df.loc[:, predictions_cols]
    predictions.to_excel(f"predictions.xlsx", index=True)

    datamodule.ids_trn = best['ids_trn']
    datamodule.ids_val = best['ids_val']

    datamodule.plot_split(f"_best_{best['fold']:04d}")

    if model_framework == "stand_alone":
        eval_loss(best['loss_info'], None, is_log=True, is_save=False, file_suffix=f"_best_{best['fold']:04d}")
        if config.model.name == "xgboost":
            best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model")
        elif config.model.name == "catboost":
            best["model"].save_model(f"epoch_{best['model'].best_iteration_}_best_{best['fold']:04d}.model")
        elif config.model.name == "lightgbm":
            best["model"].save_model(f"epoch_{best['model'].best_iteration}_best_{best['fold']:04d}.model", num_iteration=best['model'].best_iteration)
        elif config.model.name == "logistic_regression":
            pickle.dump(best["model"], open(f"logistic_regression_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        elif config.model.name == "svm":
            pickle.dump(best["model"], open(f"svm_best_{best['fold']:04d}.pkl", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Model {config.model.name} is not supported")

    y_trn = df.loc[df.index[datamodule.ids_trn], target].values
    y_trn_pred = df.loc[df.index[datamodule.ids_trn], "pred"].values
    y_trn_pred_prob = df.loc[df.index[datamodule.ids_trn], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
    y_val = df.loc[df.index[datamodule.ids_val], target].values
    y_val_pred = df.loc[df.index[datamodule.ids_val], "pred"].values
    y_val_pred_prob = df.loc[df.index[datamodule.ids_val], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
    y_tst = {}
    y_tst_pred = {}
    y_tst_pred_prob = {}
    for tst_set_name in ids_tst:
        y_tst[tst_set_name] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], target].values
        y_tst_pred[tst_set_name] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], "pred"].values
        y_tst_pred_prob[tst_set_name] = df.loc[df.index[datamodule.ids_tst[tst_set_name]], [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values

    metrics_trn = eval_classification(config, class_names, y_trn, y_trn_pred, y_trn_pred_prob, None, 'trn', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_names = metrics_trn.index.values
    metrics_trn_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['trn'])
    for metric in metrics_names:
        metrics_trn_cv.at[f"{metric}_cv_mean", 'trn'] = metrics_cv[f"trn_{metric}"].mean()
        metrics_trn_cv.at[f"{metric}_cv_std", 'trn'] = metrics_cv[f"trn_{metric}"].std()
    metrics_trn = pd.concat([metrics_trn, metrics_trn_cv])
    metrics_all = metrics_trn.copy()

    metrics_val = eval_classification(config, class_names, y_val, y_val_pred, y_val_pred_prob, None, 'val', is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
    metrics_val_cv = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=['val'])
    for metric in metrics_names:
        metrics_val_cv.at[f"{metric}_cv_mean", 'val'] = metrics_cv[f"val_{metric}"].mean()
        metrics_val_cv.at[f"{metric}_cv_std", 'val'] = metrics_cv[f"val_{metric}"].std()
    metrics_val = pd.concat([metrics_val, metrics_val_cv])
    metrics_all.loc[metrics_all.index.values, 'val'] = metrics_val.loc[metrics_all.index.values, 'val']

    metrics_tst = {}
    metrics_tst_cv = {}
    for tst_set_name in ids_tst:
        metrics_tst[tst_set_name] = eval_classification(config, class_names, y_tst[tst_set_name], y_tst_pred[tst_set_name], y_tst_pred_prob[tst_set_name], None, tst_set_name, is_log=False, is_save=True, file_suffix=f"_best_{best['fold']:04d}")
        metrics_tst_cv[tst_set_name] = pd.DataFrame(index=[f"{x}_cv_mean" for x in metrics_names] + [f"{x}_cv_std" for x in metrics_names], columns=[tst_set_name])
        for metric in metrics_names:
            metrics_tst_cv[tst_set_name].at[f"{metric}_cv_mean", tst_set_name] = metrics_cv[f"{tst_set_name}_{metric}"].mean()
            metrics_tst_cv[tst_set_name].at[f"{metric}_cv_std", tst_set_name] = metrics_cv[f"{tst_set_name}_{metric}"].std()
        metrics_tst[tst_set_name] = pd.concat([metrics_tst[tst_set_name], metrics_tst_cv[tst_set_name]])
        metrics_all.loc[metrics_all.index.values, tst_set_name] = metrics_tst[tst_set_name].loc[metrics_all.index.values, tst_set_name]

    metrics_all["trn_val"] = metrics_all.loc[:,['trn','val']].sum(axis=1) / 2
    for tst_set_name in ids_tst:
        metrics_all[f"trn_val_{tst_set_name}"] = metrics_all.loc[:, ['trn', 'val', tst_set_name]].sum(axis=1) / 3
        metrics_all[f"val_{tst_set_name}"] = metrics_all.loc[:, ['val', tst_set_name]].sum(axis=1) / 2
    metrics_all.to_excel(f"metrics_all_best_{best['fold']:04d}.xlsx", index=True, index_label="metric")

    features_labels = []
    for f in best['feature_importances']['feature'].values:
        features_labels.append(features['labels'][f])
    best['feature_importances']['feature_label'] = features_labels
    save_feature_importance(best['feature_importances'], config.num_top_features)

    expl_data = {
        'model': best["model"],
        'predict_func': best['predict_func'],
        'df': df,
        'features': features,
        'class_names': class_names,
        'target': target,
        'ids': {
            'all': np.arange(df.shape[0]),
            'trn': datamodule.ids_trn,
            'val': datamodule.ids_val,
        }
    }
    for tst_set_name in ids_tst:
        expl_data['ids'][tst_set_name] = ids_tst[tst_set_name]

    if config.is_shap == True:
        explain_shap(config, expl_data)

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return metrics_all.at[optimized_metric, config.optimized_part]
