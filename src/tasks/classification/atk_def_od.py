import numpy as np
import pandas as pd
from sympy import im
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import matplotlib.lines as mlines
from src.tasks.routines import plot_cls_dim_red, calc_confidence
from src.tasks.metrics import get_cls_pred_metrics
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import os
import seaborn as sns
import scipy.stats
from tqdm import tqdm

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.dif import DIF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.loda import LODA

from pythresh.thresholds.iqr import IQR
from pythresh.thresholds.mad import MAD
from pythresh.thresholds.fwfm import FWFM
from pythresh.thresholds.yj import YJ
from pythresh.thresholds.zscore import ZSCORE
from pythresh.thresholds.aucp import AUCP
from pythresh.thresholds.qmcd import QMCD
from pythresh.thresholds.fgd import FGD
from pythresh.thresholds.dsn import DSN
from pythresh.thresholds.clf import CLF
from pythresh.thresholds.filter import FILTER
from pythresh.thresholds.eb import EB
from pythresh.thresholds.regr import REGR
from pythresh.thresholds.hist import HIST
from pythresh.thresholds.chau import CHAU
from pythresh.thresholds.gesd import GESD
from pythresh.thresholds.mtt import MTT
from pythresh.thresholds.karch import KARCH
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.thresholds.clust import CLUST
from pythresh.thresholds.decomp import DECOMP
from pythresh.thresholds.meta import META
from pythresh.thresholds.vae import VAE
from pythresh.thresholds.cpd import CPD
from pythresh.thresholds.gamgmm import GAMGMM
from pythresh.thresholds.mixmod import MIXMOD

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular import model_sweep
import warnings
from sklearn.model_selection import train_test_split
from src.models.tabular.widedeep.tab_net import WDTabNetModel
from src.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics

from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    BasicIterativeMethod,
    MomentumIterativeMethod,
    NewtonFool,
    ZooAttack,
    ElasticNet,
    CarliniL2Method
)

log = utils.get_logger(__name__)
from sklearn.model_selection import train_test_split


def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.6, frac_val=0.15, frac_test=0.25,
                                         random_state=None):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label.
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train),
                                                          random_state=random_state)

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test,
                                                      random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


def atk_def_od_classification(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    feats_dict = datamodule.get_features()
    feats = feats_dict['all']
    n_feats = len(feats_dict['all'])
    config.in_dim = n_feats
    class_names = datamodule.get_class_names()
    classes_dict = datamodule.target_classes_dict
    classes_dict_rev = {v: k for k, v in classes_dict.items()}
    target = datamodule.target
    target_label = datamodule.target_label
    prob_cols = [f"Prob {cl_name}" for cl_name in class_names]
    if datamodule.target_classes_num != config.out_dim:
        raise ValueError(f"Inconsistent out_dim. From datamodule: {datamodule.target_classes_num}, from config: {config.out_dim}")
    df = datamodule.get_data()

    # Data parts  ======================================================================================================
    ids_dict = {
        'trn_val': df.index[df[config.datamodule.split_explicit_feat].isin(['trn', 'val', 'trn_val'])].values,
        'tst': df.index[df[config.datamodule.split_explicit_feat].isin(['tst'])].values,
        'all': df.index[df[config.datamodule.split_explicit_feat].isin(['trn', 'val', 'trn_val', 'tst'])].values,
    }

    # Setup models features  ===========================================================================================
    widedeep = datamodule.get_widedeep()
    embedding_dims = [(x[1], x[2]) for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
    categorical_cardinality = [x[1] for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else [] # type: ignore
    if config.model.name.startswith('widedeep'):
        config.model.column_idx = widedeep['column_idx']
        config.model.cat_embed_input = widedeep['cat_embed_input']
        config.model.continuous_cols = widedeep['continuous_cols']
    elif config.model.name.startswith('pytorch_tabular'):
        config.model.continuous_cols = feats_dict['con']
        config.model.categorical_cols = feats_dict['cat']
        config.model.embedding_dims = embedding_dims
        config.model.categorical_cardinality = categorical_cardinality

    # Load model =======================================================================================================
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
    model = model.to('cpu')
    model.eval()
    model.freeze()

    # Get model results for original data ==============================================================================
    model.produce_probabilities = True
    y_pred_prob = model(torch.from_numpy(np.float32(df.loc[:, feats].values))).cpu().detach().numpy()
    y_pred = np.argmax(y_pred_prob, 1)
    df["Prediction"] = y_pred
    for cl_name, cl_id in classes_dict.items():
        df[f"Prob {cl_name}"] = y_pred_prob[:, cl_id]

    # Save original data ===============================================================================================
    df.to_excel("df.xlsx", index=True)

    # Calc metrics' confidence intervals ===============================================================================
    Path(f"Origin/confidence").mkdir(parents=True, exist_ok=True)
    metrics = get_cls_pred_metrics(len(class_names))
    calc_confidence(df, target, 'Prediction', metrics, f"Origin/confidence", task="classification")

    # Plot model =======================================================================================================
    df_fig = df.loc[:, [f"Prob {cl_name}" for cl_name in class_names] + [target]].copy()
    df_fig[target].replace(classes_dict_rev, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df_fig,
        x=f"Prob {class_names[-1]}",
        hue=target,
        linewidth=2,
        cut=0,
        fill=True,
        ax=ax
    )
    plt.savefig(f"kde_proba.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"kde_proba.pdf", bbox_inches='tight')
    plt.close(fig)
    for data_part, ids in ids_dict.items():
        cm = confusion_matrix(df.loc[ids, target].values, df.loc[ids, 'Prediction'].values)
        if len(cm) > 1:
            cm_sum = np.sum(cm, axis=1, keepdims=True)
            cm_perc = cm / cm_sum.astype(float) * 100
            annot = np.empty_like(cm).astype(str)
            nrows, ncols = cm.shape
            for i in range(nrows):
                for j in range(ncols):
                    c = cm[i, j]
                    p = cm_perc[i, j]
                    if i == j:
                        s = cm_sum[i]
                        annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                    elif c == 0:
                        annot[i, j] = ''
                    else:
                        annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm.index.name = 'Actual'
            cm.columns.name = 'Predicted'
            fig, ax = plt.subplots(figsize=(2 * len(class_names), 2 * len(class_names)))
            sns.heatmap(cm, annot=annot, fmt='', ax=ax)
            plt.savefig(f"confusion_matrix_{data_part}.png", bbox_inches='tight')
            plt.savefig(f"confusion_matrix_{data_part}.pdf", bbox_inches='tight')
            plt.close(fig)

    # Outliers methods =================================================================================================
    if config.outliers:
        classifiers = {
            'ECDF-Based (ECOD)': ECOD(),
            'Copula-Based (COPOD)': COPOD(),
            'Principal Component Analysis (PCA)': PCA(),
            'Local Outlier Factor (LOF)': LOF(),
            'Connectivity-Based Outlier Factor (COF)': COF(),
            'Clustering-Based Local Outlier Factor (CBLOF)': CBLOF(),
            'Histogram-based Outlier Score (HBOS)': HBOS(),
            'k Nearest Neighbors (kNN)': KNN(),
            'Subspace Outlier Detection (SOD)': SOD(),
            'Isolation Forest': IForest(),
            'Isolation-Based with Nearest-Neighbor Ensembles (INNE)': INNE(),
            'Deep Isolation Forest for Anomaly Detection (DIF)': DIF(),
            'Feature Bagging': FeatureBagging(),
            'Lightweight On-line Detector of Anomalies (LODA)': LODA(),
        }

        thresholders = {
            "Inter-Quartile Region (IQR)": IQR(),
            "Median Absolute Deviation (MAD)": MAD(),
            "Full Width at Full Minimum (FWFM)": FWFM(),
            "Yeo-Johnson Transformation (YJ)": YJ(),
            "Z Score (ZSCORE)": ZSCORE(),
            "AUC Percentage (AUCP)": AUCP(),
            "Quasi-Monte Carlo Discreperancy (QMCD)": QMCD(),
            "Fixed Gradient Descent (FGD)": FGD(),
            "Distance Shift from Normal (DSN)": DSN(),
            "Trained Classifier (CLF)": CLF(),
            "Filtering Based (FILTER)": FILTER(),
            "Elliptical Boundary (EB)": EB(),
            "Regression Intercept (REGR)": REGR(),
            "Histogram Based Methods (HIST)": HIST(),
            "Chauvenet's Criterion (CHAU)": CHAU(),
            "Generalized Extreme Studentized Deviate (GESD)": GESD(),
            "Modified Thompson Tau Test (MTT)": MTT(),
            "Karcher Mean (KARCH)": KARCH(),
            "One-Class SVM (OCSVM)": OCSVM(),
            "Clustering (CLUST)": CLUST(),
            "Decomposition (DECOMP)": DECOMP(),
            "Meta-model (META)": META(),
            "Variational Autoencoder (VAE)": VAE(),
            "Change Point Detection (CPD)": CPD(),
            "Bayesian Gamma GMM (GAMGMM)": GAMGMM(skip=True),
            "Mixture Models (MIXMOD)": MIXMOD(),
        }
        
        # Calc outliers for original data ==============================================================================
        log.info("Calc outliers for original data")
        df_outs = pd.DataFrame(index=list(classifiers.keys()), columns=list(thresholders.keys()))
        for pyod_m_name, pyod_m in classifiers.items():
            log.info(pyod_m_name)
            scores = pyod_m.fit(df.loc[ids_dict['tst'], feats].values).decision_scores_
            for pythresh_m_name, pythresh_m in thresholders.items():
                labels = pythresh_m.eval(scores)
                df_outs.at[pyod_m_name, pythresh_m_name] = sum(labels) / len(labels) * 100
        df_outs.to_excel(f"outliers.xlsx")
        
        df_fig = df_outs.astype(float)
        sns.set_theme(style='ticks', font_scale=1.0)
        fig, ax = plt.subplots(figsize=(16, 7))
        heatmap = sns.heatmap(
            df_fig,
            annot=True,
            fmt=".1f",
            cmap='hot',
            linewidth=0.1,
            linecolor='black',
            cbar_kws={
                'orientation': 'horizontal',
                'location': 'top',
                'pad': 0.025,
                'aspect': 30
            },
            annot_kws={"size": 12},
            ax=ax
        )
        ax.set_ylabel('Outliers Detection Algorithms')
        ax.set_xlabel('Thresholding Algorithms')
        heatmap_pos = heatmap.get_position()
        ax.figure.axes[-1].set_title("Outliers' percentage")
        ax.figure.axes[-1].tick_params()
        for spine in ax.figure.axes[-1].spines.values():
            spine.set_linewidth(1)
        plt.savefig(f"outliers.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"outliers.pdf", bbox_inches='tight')
        plt.close(fig)

    # Attacks ==========================================================================================================
    if config.attacks:
        df['Data'] = 'Origin'

        ids_atk = ids_dict['tst']

        art_classifier = PyTorchClassifier(
            model=model,
            loss=model.loss_fn,
            input_shape=(len(feats),),
            nb_classes=2,
            optimizer=torch.optim.Adam(
                params=model.parameters(),
                lr=model.hparams.optimizer_lr,
                weight_decay=model.hparams.optimizer_weight_decay
            ),
            use_amp=False,
            opt_level="O1",
            loss_scale="dynamic",
            channels_first=True,
            clip_values=(0.0, 1.0),
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0.0, 1.0),
            device_type="cpu"
        )

        atk_types = {
            "Eps": {
                'names': ['MomentumIterative', 'BasicIterative', 'FastGradient'],
                'values': sorted(list(set.union(
                    set(np.linspace(0.1, 1.0, 10)),
                    set(np.linspace(0.01, 0.1, 10)),
                    set(np.linspace(0.001, 0.01, 10))
                ))),
                'colors': {
                    "MomentumIterative": px.colors.qualitative.D3[0],
                    "BasicIterative": px.colors.qualitative.D3[1],
                    "FastGradient": px.colors.qualitative.D3[3],
                }
            },
            "BSS": {
                'names': ['ElasticNet', 'CarliniL2Method', 'ZooAttack'],
                'values': list(range(1, 11, 1)),
                'colors': {
                    "ElasticNet": px.colors.qualitative.G10[7],
                    "CarliniL2Method": px.colors.qualitative.G10[8],
                    "ZooAttack": px.colors.qualitative.G10[9],
                }
            },
            "Eta": {
                'names': ['NewtonFool'],
                'values': sorted(list(set.union(
                    set(np.linspace(0.1, 1.0, 10)),
                    set(np.linspace(0.01, 0.1, 10)),
                    set(np.linspace(0.001, 0.01, 10))
                ))),
                'colors': {
                    'NewtonFool': px.colors.qualitative.T10[7],
                }
            }
        }

        df_ori = df.loc[ids_atk, :].copy()

        for atk_type, atk_dict in atk_types.items():
            for val in atk_dict['values']:
                if atk_type == "Eps":
                    eps = np.array([val * scipy.stats.iqr(df.loc[ids_atk, feat].values) for feat in feats])
                    eps_step = np.array([0.2 * val * scipy.stats.iqr(df.loc[ids_atk, feat].values) for feat in feats])
                    attacks = {
                        'MomentumIterative': MomentumIterativeMethod(
                            estimator=art_classifier,
                            norm=np.inf,
                            eps=eps,
                            eps_step=eps_step,
                            decay=0.1,
                            max_iter=100,
                            targeted=False,
                            batch_size=512,
                            verbose=True
                        ),
                        'BasicIterative': BasicIterativeMethod(
                            estimator=art_classifier,
                            eps=eps,
                            eps_step=eps_step,
                            max_iter=100,
                            targeted=False,
                            batch_size=512,
                            verbose=True
                        ),
                        'FastGradient': FastGradientMethod(
                            estimator=art_classifier,
                            norm=np.inf,
                            eps=eps,
                            eps_step=eps_step,
                            targeted=False,
                            num_random_init=0,
                            batch_size=512,
                            minimal=False,
                            summary_writer=False,
                        ),
                    }
                    label_suffix = f'Eps: {val:0.4f}'
                elif atk_type == "BSS":
                    attacks = {
                        'ElasticNet': ElasticNet(
                            classifier=art_classifier,
                            confidence=0.0,
                            targeted=False,
                            learning_rate=1e-3,
                            binary_search_steps=val,
                            max_iter=20,
                            beta=1e-3,
                            initial_const=1e-4,
                            batch_size=1,
                            decision_rule="EN",
                            verbose=True,
                        ),
                        'CarliniL2Method': CarliniL2Method(
                            classifier=art_classifier,
                            confidence=0.0,
                            targeted=False,
                            learning_rate=0.001,
                            binary_search_steps=val,
                            max_iter=20,
                            initial_const=1e-4,
                            max_halving=5,
                            max_doubling=5,
                            batch_size=1,
                            verbose=True
                        ),
                        'ZooAttack': ZooAttack(
                            classifier=art_classifier,
                            confidence=0.0,
                            targeted=False,
                            learning_rate=0.001,
                            max_iter=20,
                            binary_search_steps=val,
                            initial_const=1e-4,
                            abort_early=True,
                            use_resize=False,
                            use_importance=True,
                            nb_parallel=16,
                            batch_size=1,
                            variable_h=0.001,
                            verbose=True
                        ),
                    }
                    label_suffix = f'BSS: {val}'
                elif atk_type == "Eta":
                    attacks = {
                        'NewtonFool': NewtonFool(
                            classifier=art_classifier,
                            max_iter=100,
                            eta=val,
                            batch_size=100,
                            verbose=True,
                        ),
                    }
                    label_suffix = f'Eta: {val:0.2e}'
                else:
                    raise ValueError("Unsupported atk_type")

                for atk_name, atk in attacks.items():
                    if atk_type == "Eps":
                        path_curr = f"Evasion/{atk_name}/eps_{val:0.4f}"
                    elif atk_type == "BSS":
                        path_curr = f"Evasion/{atk_name}/bss_{val}"
                    elif atk_type == "Eta":
                        path_curr = f"Evasion/{atk_name}/eta_{val:0.2e}"
                    else:
                        raise ValueError("Unsupported atk_type")
                    Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

                    X_atk = atk.generate(np.float32(df.loc[ids_atk, feats].values))

                    df_atk = df.loc[ids_atk, [target]].copy()
                    df_atk.loc[ids_atk, feats] = X_atk

                    model.produce_probabilities = True
                    y_pred_prob = model(torch.from_numpy(np.float32(df_atk.loc[:, feats].values))).cpu().detach().numpy()
                    y_pred = np.argmax(y_pred_prob, 1)
                    df_atk.loc[:, "Prediction"] = y_pred
                    for cl_name, cl_id in classes_dict.items():
                        df_atk.loc[:, f"Prob {cl_name}"] = y_pred_prob[:, cl_id]
                    df_atk['Data'] = f'{atk_name} {label_suffix}'
                    df_atk.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')

                    # Calc metrics' ====================================================================================
                    Path(f"{path_curr}/confidence").mkdir(parents=True, exist_ok=True)
                    metrics = get_cls_pred_metrics(len(class_names))
                    calc_confidence(df_atk, target, 'Prediction', metrics, f"{path_curr}/confidence", task="classification")

            # Plot attacks' metrics from param values ==================================================================
            metrics_names = {
                'accuracy_weighted': 'Accuracy',
            }
            for m in metrics_names:
                df_params = pd.DataFrame(index=atk_dict['values'])
                for val in atk_dict['values']:
                    for atk_name in atk_dict['names']:
                        if atk_type == "Eps":
                            path_curr = f"Evasion/{atk_name}/eps_{val:0.4f}"
                        elif atk_type == "BSS":
                            path_curr = f"Evasion/{atk_name}/bss_{val}"
                        elif atk_type == "Eta":
                            path_curr = f"Evasion/{atk_name}/eta_{val:0.2e}"
                        else:
                            raise ValueError("Unsupported atk_type")
                        df_metrics = pd.read_excel(f"{path_curr}/confidence/metrics.xlsx", index_col=0)
                        df_params.at[val, atk_name] = df_metrics.at[m, "value"]
                df_params[atk_type] = df_params.index.values
                df_fig = df_params.melt(id_vars=atk_type, var_name='Method', value_name=metrics_names[m])
                fig = plt.figure()
                sns.set_theme(style='whitegrid', font_scale=1)
                lines = sns.lineplot(
                    data=df_fig,
                    x=atk_type,
                    y=metrics_names[m],
                    hue=f"Method",
                    style=f"Method",
                    palette=atk_dict['colors'],
                    hue_order=atk_dict['names'],
                    markers=True,
                    dashes=False,
                )
                if atk_type in ["Eps", 'Eta']:
                    plt.xscale('log')
                x_ptp = np.ptp(atk_dict['values'])
                x_min = np.min(atk_dict['values']) - 0.1 * x_ptp
                x_max = np.max(atk_dict['values']) + 0.1 * x_ptp
                metrics_base = pd.read_excel(f"Origin/confidence/metrics.xlsx", index_col=0).at[m, "value"]
                lines.set_xlim(x_min, x_max)
                plt.gca().plot(
                    [x_min, x_max],
                    [metrics_base, metrics_base],
                    color='k',
                    linestyle='dashed',
                    linewidth=1
                )
                plt.savefig(f"Evasion/{m}_vs_{atk_type}.png", bbox_inches='tight', dpi=200)
                plt.savefig(f"Evasion/{m}_vs_{atk_type}.pdf", bbox_inches='tight')
                plt.close(fig)
    
    # Outliers for attacks =============================================================================================    
    if config.outliers:
        attacks_options = {
            'Eps': {
                'types': ['MomentumIterative', 'BasicIterative', 'FastGradient'],
                'values': [0.005, 0.02, 0.05, 0.2]
            },
            'BSS': {
                'types': ['ElasticNet', 'CarliniL2Method', 'ZooAttack'],
                'values': [2, 4, 6, 8]
            },
            'Eta': {
                'types': ['NewtonFool'],
                'values': [1e-3, 2e-3, 3e-3, 1e-2]
            },
        }

        for var_param, opt in attacks_options.items():
            print(var_param)
            for atk_type in opt['types']:
                print(atk_type)
                for var_val in opt['values']:
                    print(var_val)
                    if var_param == 'Eps':
                        path_curr = f"Evasion/{atk_type}/eps_{var_val:0.4f}"
                    elif var_param == 'BSS':
                        path_curr = f"Evasion/{atk_type}/bss_{var_val}"
                    else:
                        path_curr = f"Evasion/{atk_type}/eta_{var_val:0.2e}"
                        
                    df_adv = pd.read_excel(f"{path_curr}/df.xlsx", index_col='sample_id')
                    
                    df_outs = pd.DataFrame(index=list(classifiers.keys()), columns=list(thresholders.keys()))
                    for pyod_m_name, pyod_m in classifiers.items():
                        scores = pyod_m.fit(df_adv.loc[:, feats].values).decision_scores_
                        for pythresh_m_name, pythresh_m in thresholders.items(): 
                            labels = pythresh_m.eval(scores)
                            df_outs.at[pyod_m_name, pythresh_m_name] = sum(labels) / len(labels) * 100
                            
                    df_outs.to_excel(f"{path_curr}/outliers.xlsx")
                    
                    df_fig = df_outs.astype(float)
                    sns.set_theme(style='ticks', font_scale=1.0)
                    fig, ax = plt.subplots(figsize=(16, 7))
                    heatmap = sns.heatmap(
                        df_fig,
                        annot=True,
                        fmt=".1f",
                        cmap='hot',
                        linewidth=0.1,
                        linecolor='black',
                        cbar_kws={
                            'orientation': 'horizontal',
                            'location': 'top',
                            'pad': 0.025,
                            'aspect': 30
                        },
                        annot_kws={"size": 12},
                        ax=ax
                    )
                    ax.set_ylabel('Outliers Detection Algorithms')
                    ax.set_xlabel('Thresholding Algorithms')
                    heatmap_pos = heatmap.get_position()
                    ax.figure.axes[-1].set_title("Outliers' percentage")
                    ax.figure.axes[-1].tick_params()
                    for spine in ax.figure.axes[-1].spines.values():
                        spine.set_linewidth(1)
                    plt.savefig(f"{path_curr}/outliers.png", bbox_inches='tight', dpi=200)
                    plt.savefig(f"{path_curr}/outliers.pdf", bbox_inches='tight')
                    plt.close(fig)

    # Defences =========================================================================================================
    if config.defence:
        df_ori = df.loc[ids_atk, feats].copy()
        df_ori['Class'] = 'Original'

        epsilons = sorted(list(set.union(
            set(np.linspace(0.1, 1.0, 10)), 
            set(np.linspace(0.01, 0.1, 10)),
            set(np.linspace(0.001, 0.01, 10))
        )))
        bsss = list(range(1, 11, 1))
        etas = sorted(list(set.union(
            set(np.linspace(0.1, 1.0, 10)), 
            set(np.linspace(0.01, 0.1, 10)),
            set(np.linspace(0.001, 0.01, 10))
        )))

        attacks_options = {
            'Eps': {
                'types': ['MomentumIterative', 'BasicIterative', 'FastGradient'],
                'values': epsilons
            },
            'BSS': {
                'types': ['ElasticNet', 'CarliniL2Method', 'ZooAttack'],
                'values': bsss
            },
            'Eta': {
                'types': ['NewtonFool'],
                'values': etas
            },
        }
        
        datasets = {}
        for var_param, opt in attacks_options.items():
            print(var_param)
            datasets[var_param] = {}
            for atk_type in opt['types']:
                print(atk_type)
                datasets[var_param][atk_type] = {}
                for var_val_id, var_val in enumerate(opt['values']):
                    if var_param == 'Eps':
                        path_curr = f"Evasion/{atk_type}/eps_{var_val:0.4f}"
                        val_str = f'{var_val:0.4f}'
                    elif var_param == 'BSS':
                        path_curr = f"Evasion/{atk_type}/bss_{var_val}"
                        val_str = f'{var_val}'
                    else:
                        path_curr = f"Evasion/{atk_type}/eta_{var_val:0.2e}"
                        val_str = f'{var_val:0.2e}'
                    print(val_str)
                    df_adv = pd.read_excel(f"{path_curr}/df.xlsx", index_col=0)
                    df_adv = df_adv[feats]
                    df_adv['Class'] = 'Attack'
                    datasets[var_param][atk_type][val_str] = df_adv

        for var_param, opt in attacks_options.items():
            print(var_param)
            for atk_type in opt['types']:
                print(atk_type)
                
                df_def_acc = pd.DataFrame(index=opt['values'], columns=['Model'] + list(opt['values']))
                
                for var_val_id, var_val in enumerate(opt['values']):
                    print(var_val)
                    if var_param == 'Eps':
                        path_curr = f"Evasion/{atk_type}/eps_{var_val:0.4f}"
                        val_str = f'{var_val:0.4f}'
                    elif var_param == 'BSS':
                        path_curr = f"Evasion/{atk_type}/bss_{var_val}"
                        val_str = f'{var_val}'
                    else:
                        path_curr = f"Evasion/{atk_type}/eta_{var_val:0.2e}"
                        val_str = f'{var_val:0.2e}'
                        
                    df_adv = datasets[var_param][atk_type][val_str]
                    df_def = pd.concat([df_ori.loc[ids_atk, :], df_adv.loc[ids_atk, :]])
                    
                    df_def_trn, df_def_val, df_def_tst = split_stratified_into_train_val_test(
                        df_def, 
                        stratify_colname='Class',
                        frac_train=0.60,
                        frac_val=0.20,
                        frac_test=0.20
                    )
                    
                    data_config = DataConfig(
                        target=['Class'],
                        continuous_cols=list(feats),
                        continuous_feature_transform='yeo-johnson',
                        normalize_continuous_features=True,
                    )
                    
                    trainer_config = TrainerConfig(
                        batch_size=1024,
                        max_epochs=100,
                        min_epochs=1,
                        auto_lr_find=True,
                        early_stopping='valid_loss',
                        early_stopping_min_delta=0.0001,
                        early_stopping_mode='min',
                        early_stopping_patience=100,
                        checkpoints='valid_loss',
                        checkpoints_path=f"{path_curr}/detector",
                        load_best=True,
                        progress_bar='none',
                        seed=42
                    )
                    
                    optimizer_config = OptimizerConfig(
                        optimizer='Adam',
                        lr_scheduler='CosineAnnealingWarmRestarts',
                        lr_scheduler_params={
                            'T_0': 10,
                            'T_mult': 1,
                            'eta_min': 0.00001,
                        },
                        lr_scheduler_monitor_metric='valid_loss'
                    )
            
                    head_config = LinearHeadConfig(
                        layers='',
                        activation='ReLU',
                        dropout=0.1,
                        use_batch_norm=False,
                        initialization='xavier',
                    ).__dict__
                    
                    model_config = CategoryEmbeddingModelConfig(
                        task="classification",
                        layers="256-128-64",
                        activation="LeakyReLU",
                        dropout=0.1,
                        initialization="kaiming",
                        head="LinearHead",
                        head_config=head_config,
                        learning_rate=1e-3,
                    )
                    
                    # model_list = [model_config]
                    model_list = 'lite'
            
                    sweep_df, best_model = model_sweep(
                        task="classification",
                        train=df_def_trn,
                        validation=df_def_val,
                        test=df_def_tst,
                        data_config=data_config,
                        optimizer_config=optimizer_config,
                        trainer_config=trainer_config,
                        model_list=model_list,
                        common_model_args=dict(head="LinearHead", head_config=head_config),
                        metrics=[
                            'accuracy',
                            'f1_score',
                            'precision',
                            'recall',
                            'specificity',
                            'cohen_kappa',
                            'auroc'
                        ],
                        metrics_prob_input=[True, True, True, True, True, True, True],
                        metrics_params=[
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                            {'task': 'multiclass', 'num_classes': 2},
                            {'task': 'multiclass', 'num_classes': 2, 'average': 'weighted'},
                        ],
                        rank_metric=("accuracy", "higher_is_better"),
                        progress_bar=False,
                        verbose=False,
                        suppress_lightning_logger=True,
                    )
                    
                    ckpts = glob(f"{path_curr}/detector/*")
                    for ckpt in ckpts:
                        os.remove(ckpt)
                    # best_model.save_model(f"{path_curr}/detector")
                    df_def_acc.at[var_val, 'Model'] = best_model.config['_model_name']
                    
                    for tst_var_val_id, tst_var_val in enumerate(opt['values']):
                        if tst_var_val != var_val:
                            print(f"Testing: {tst_var_val}")
                            if var_param == 'Eps':
                                path_tst = f"Evasion/{atk_type}/eps_{tst_var_val:0.4f}"
                                val_str_tst = f'{tst_var_val:0.4f}'
                            elif var_param == 'BSS':
                                path_tst = f"Evasion/{atk_type}/bss_{tst_var_val}"
                                val_str_tst = f'{tst_var_val}'
                            else:
                                path_tst = f"Evasion/{atk_type}/eta_{tst_var_val:0.2e}"
                                val_str_tst = f'{tst_var_val:0.2e}'

                            df_adv_tst = datasets[var_param][atk_type][val_str_tst]

                            df_def_tst_eps = pd.concat([df_ori, df_adv_tst])
                            metrics = best_model.evaluate(test=df_def_tst_eps, verbose=False)[0]
                            df_def_acc.at[var_val, tst_var_val] = metrics['test_accuracy']
                df_def_acc.to_excel(f"Evasion/{atk_type}/detectors_accuracy.xlsx")
                
        for var_param, opt in attacks_options.items():
            print(var_param)

            for atk_type in opt['types']:
                print(atk_type)
                df_def_acc = pd.read_excel(f"Evasion/{atk_type}/detectors_accuracy.xlsx", index_col=0)
                df_def_acc.drop(['Model'], axis=1, inplace=True)

                if var_param == 'Eps':
                    figsize=(13, 12)
                    df_def_acc.rename(columns={x: f"{x:.3f}" for x in df_def_acc.columns}, inplace=True)
                    df_def_acc['index'] = [f"{x:.3f}" for x in df_def_acc.index.values]
                elif var_param == 'BSS':
                    figsize=(6, 6)
                    df_def_acc.rename(columns={x: f"{x:d}" for x in df_def_acc.columns}, inplace=True)
                    df_def_acc['index'] = [f"{x:d}" for x in df_def_acc.index.values]
                else:
                    figsize=(13, 12)
                    df_def_acc.rename(columns={x: f"{x:.4f}" for x in df_def_acc.columns}, inplace=True)
                    df_def_acc['index'] = [f"{x:.4f}" for x in df_def_acc.index.values]
                df_def_acc.set_index('index', inplace=True)
                
                df_fig = df_def_acc.astype(float)
                sns.set_theme(style='ticks', font_scale=1.0)
                fig, ax = plt.subplots(figsize=figsize)
                heatmap = sns.heatmap(
                    df_fig,
                    annot=True,
                    fmt=".2f",
                    cmap='hot',
                    linewidth=0.1,
                    linecolor='black',
                    cbar_kws={
                        'orientation': 'horizontal',
                        'location': 'top',
                        'pad': 0.025,
                        'aspect': 30
                    },
                    annot_kws={"size": 9},
                    ax=ax
                )
                ax.set_xlabel('Test Attack Strength')
                ax.set_ylabel('Train Attack Strength')
                heatmap_pos = heatmap.get_position()
                ax.figure.axes[-1].set_title("Accuracy")
                ax.figure.axes[-1].tick_params()
                for spine in ax.figure.axes[-1].spines.values():
                    spine.set_linewidth(1)
                plt.savefig(f"Evasion/{atk_type}/detectors_accuracy.png", bbox_inches='tight', dpi=200)
                plt.savefig(f"Evasion/{atk_type}/detectors_accuracy.pdf", bbox_inches='tight')
                plt.close(fig)