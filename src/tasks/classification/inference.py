import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import xgboost as xgb
from src.tasks.routines import eval_classification
from catboost import CatBoost
import plotly.express as px
from src.tasks.classification.shap import explain_shap
from src.models.tabular.base import get_model_framework_dict


log = utils.get_logger(__name__)

def inference_classification(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    model_framework_dict = get_model_framework_dict()
    model_framework = model_framework_dict[config.model.name]

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
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

    df = df[df[config.data_part_column].notna()]
    data_parts = df[config.data_part_column].dropna().unique()
    data_part_main = config.data_part_main
    data_parts = [data_part_main] + list(set(data_parts) - set([data_part_main]))
    indexes = {}
    X = {}
    y = {}
    y_pred = {}
    y_pred_prob = {}
    y_pred_raw = {}
    colors = {}
    for data_part_id, data_part in enumerate(data_parts):
        indexes[data_part] = df.loc[df[config.data_part_column] == data_part, :].index.values
        X[data_part] = df.loc[indexes[data_part], features['all']].values
        y[data_part] = df.loc[indexes[data_part], target].values
        colors[data_part] = px.colors.qualitative.Light24[data_part_id]

    if model_framework == "pytorch":
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
            num_unique_vals = [len(np.unique(X[data_part_main][:, i])) for i in range(X[data_part_main].shape[1])]
            num_units = [min(config.model.num_basis_functions, i * config.model.units_multiplier) for i in num_unique_vals]
            config.model.num_units = num_units
        log.info(f"Instantiating model <{config.model._target_}>")
        model = hydra.utils.instantiate(config.model)

        model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
        model.eval()
        model.freeze()

        model.produce_probabilities = True
        for data_part in data_parts:
            y_pred_prob[data_part] = model(torch.from_numpy(X[data_part])).cpu().detach().numpy()
        model.produce_probabilities = False
        for data_part in data_parts:
            y_pred_raw[data_part] = model(torch.from_numpy(X[data_part])).cpu().detach().numpy()
            y_pred[data_part] = np.argmax(y_pred_prob[data_part], 1)
        model.produce_probabilities = True

        def predict_func(X):
            model.produce_probabilities = True
            batch = {
                'all': torch.from_numpy(np.float32(X[:, features['all_ids']])),
                'continuous': torch.from_numpy(np.float32(X[:, features['con_ids']])),
                'categorical': torch.from_numpy(np.int32(X[:, features['cat_ids']])),
            }
            tmp = model(batch)
            return tmp.cpu().detach().numpy()

    elif model_framework == "stand_alone":

        if config.model.name == "xgboost":
            model = xgb.Booster()
            model.load_model(config.path_ckpt)
            for data_part in data_parts:
                dmat = xgb.DMatrix(X[data_part], y[data_part], feature_names=features['all'], enable_categorical=True)
                y_pred_prob[data_part] = model.predict(dmat)
                y_pred_raw[data_part] = model.predict(dmat, output_margin=True)
                y_pred[data_part] = np.argmax(y_pred_prob[data_part], 1)

            def predict_func(X):
                X = xgb.DMatrix(X, feature_names=features['all'], enable_categorical=True)
                y = model.predict(X)
                return y

        elif config.model.name == "catboost":
            model = CatBoost()
            model.load_model(config.path_ckpt)
            for data_part in data_parts:
                y_pred_prob[data_part] = model.predict(X[data_part], prediction_type="Probability")
                y_pred_raw[data_part] = model.predict(X[data_part], prediction_type="RawFormulaVal")
                y_pred[data_part] = np.argmax(y_pred_prob[data_part], 1)

            def predict_func(X):
                X = pd.DataFrame(data=X, columns=features["all"])
                X[features["cat"]] = X[features["cat"]].astype('int32')
                y = model.predict(X)
                return y

        elif config.model.name == "lightgbm":
            model = lgb.Booster(model_file=config.path_ckpt)
            for data_part in data_parts:
                y_pred_prob[data_part] = model.predict(X[data_part], num_iteration=model.best_iteration)
                y_pred_raw[data_part] = model.predict(X[data_part], num_iteration=model.best_iteration, raw_score=True)
                y_pred[data_part] = np.argmax(y_pred_prob[data_part], 1)
            def predict_func(X):
                y = model.predict(X, num_iteration=model.best_iteration)
                return y

        else:
            raise ValueError(f"Model {config.model.name} is not supported")

    else:
        raise ValueError(f"Unsupported model_framework: {model_framework}")

    for data_part in data_parts:
        df.loc[indexes[data_part], "pred"] = y_pred[data_part]
        for cl_id, cl in enumerate(class_names):
            df.loc[indexes[data_part], f"pred_prob_{cl_id}"] = y_pred_prob[data_part][:, cl_id]
            df.loc[indexes[data_part], f"pred_raw_{cl_id}"] = y_pred_raw[data_part][:, cl_id]
        eval_classification(config, class_names, y[data_part], y_pred[data_part], y_pred_prob[data_part], None, data_part, is_log=False, is_save=True, file_suffix=f"")

    df['ids'] = np.arange(df.shape[0])
    ids = {}
    for data_part in data_parts:
        ids[data_part] = df.loc[indexes[data_part], 'ids'].values
    ids['all'] = df['ids']

    if config.is_shap == True:
        shap_data = {
            'model': model,
            'predict_func': predict_func,
            'df': df,
            'features': features,
            'class_names': class_names,
            'target': target,
            'ids': ids
        }
        explain_shap(config, shap_data)

    df.to_excel("df.xlsx", index=True)
