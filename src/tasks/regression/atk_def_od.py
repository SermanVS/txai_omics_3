import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import matplotlib.pyplot as plt
from scipy.stats import iqr
import seaborn as sns
import plotly.express as px
from src.tasks.metrics import get_reg_metrics
from pathlib import Path
from tqdm import tqdm

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.qmcd import QMCD as QMCDOD
from pyod.models.sampling import Sampling
from pyod.models.gmm import GMM
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.cd import CD
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.loda import LODA
from pyod.models.lunar import LUNAR

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
from pythresh.thresholds.wind import WIND
from pythresh.thresholds.eb import EB
from pythresh.thresholds.regr import REGR
from pythresh.thresholds.mcst import MCST
from pythresh.thresholds.moll import MOLL
from pythresh.thresholds.chau import CHAU
from pythresh.thresholds.gesd import GESD
from pythresh.thresholds.karch import KARCH
from pythresh.thresholds.ocsvm import OCSVM
from pythresh.thresholds.clust import CLUST
from pythresh.thresholds.decomp import DECOMP
from pythresh.thresholds.meta import META
from pythresh.thresholds.vae import VAE
from pythresh.thresholds.cpd import CPD
from pythresh.thresholds.gamgmm import GAMGMM
from pythresh.thresholds.mixmod import MIXMOD

from art.estimators.regression.pytorch import PyTorchRegressor
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular import model_sweep

from glob import glob
import os


log = utils.get_logger(__name__)


def atk_def_od_regression(config: DictConfig):

    if "seed" in config:
        seed_everything(config.seed)

    if 'wandb' in config.logger:
        config.logger.wandb["project"] = config.project_name

    # Init Lightning datamodule  =======================================================================================
    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: TabularDataModule = hydra.utils.instantiate(config.datamodule)
    feats_dict = datamodule.get_features()
    feats = feats_dict['all']
    n_feats = len(feats_dict['all'])
    config.in_dim = n_feats
    target = datamodule.target
    target_label = datamodule.target_label
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
    categorical_cardinality = [x[1] for x in widedeep['cat_embed_input']] if widedeep['cat_embed_input'] else []
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
    log.info(f"Load model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
    model = model.to('cpu')
    model.eval()
    model.freeze()

    # Get model results for original data ==============================================================================
    log.info("Get model results for original data")
    df["Prediction"] = model(torch.from_numpy(np.float32(df.loc[:, feats].values))).cpu().detach().numpy().ravel()
    df["Error"] = df["Prediction"] - df[target]
    df["abs(Error)"] = df["Error"].abs()
    df['Data'] = 'Origin'
    
    # Save original data ===============================================================================================
    df.to_excel("df.xlsx", index=True)

    # Plot model =======================================================================================================
    fig = plt.figure()
    sns.set_theme(style='whitegrid')
    xy_min = df[[target, 'Prediction']].min().min()
    xy_max = df[[target, 'Prediction']].max().max()
    xy_ptp = xy_max - xy_min
    scatter = sns.scatterplot(
        data=df,
        x=target,
        y="Prediction",
        hue=config.datamodule.split_explicit_feat,
        linewidth=0.3,
        alpha=0.75,
        edgecolor="k",
        s=25,
    )
    scatter.set_xlabel(target)
    scatter.set_ylabel("Prediction")
    scatter.set_xlim(xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp)
    scatter.set_ylim(xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp)
    plt.gca().plot(
        [xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp],
        [xy_min - 0.1 * xy_ptp, xy_max + 0.1 * xy_ptp],
        color='k',
        linestyle='dashed',
        linewidth=1
    )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"scatter.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"scatter.pdf", bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df,
        x=config.datamodule.split_explicit_feat,
        y='Error',
        scale='width',
        saturation=0.75,
    )
    plt.savefig(f"violin.png", bbox_inches='tight', dpi=400)
    plt.savefig(f"violin.pdf", bbox_inches='tight')
    plt.close(fig)

    # Outliers methods =================================================================================================
    if config.outliers:
        classifiers = {
            'ECDF-Based (ECOD)': ECOD(),
            'Copula-Based (COPOD)': COPOD(),
            'Quasi-Monte Carlo Discrepancy (QMCD)': QMCDOD(),
            'Rapid distance-based via Sampling': Sampling(),
            'Probabilistic Mixture Modeling (GMM)': GMM(),
            'Principal Component Analysis (PCA)': PCA(),
            'Minimum Covariance Determinant (MCD)': MCD(),
            'Cook\'s Distance (CD)': CD(),
            'Deviation-based Outlier Detection (LMDD)': LMDD(),
            'Local Outlier Factor (LOF)': LOF(),
            'Connectivity-Based Outlier Factor (COF)': COF(),
            'Clustering-Based Local Outlier Factor (CBLOF)': CBLOF(),
            'Histogram-based Outlier Score (HBOS)': HBOS(),
            'k Nearest Neighbors (kNN)': KNN(),
            'Subspace Outlier Detection (SOD)': SOD(),
            'Isolation Forest': IForest(),
            'Deep Isolation Forest for Anomaly Detection (DIF)': DIF(),
            'Feature Bagging': FeatureBagging(),
            'Lightweight On-line Detector of Anomalies (LODA)': LODA(),
            'LUNAR': LUNAR()
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
            "Topological Winding Number (WIND)": WIND(),
            "Elliptical Boundary (EB)": EB(),
            "Regression Intercept (REGR)": REGR(),
            "Monte Carlo Statistical Tests (MCST)": MCST(),
            "Mollifier (MOLL)": MOLL(),
            "Chauvenet's Criterion (CHAU)": CHAU(),
            "Generalized Extreme Studentized Deviate (GESD)": GESD(),
            "Karcher Mean (KARCH)": KARCH(),
            "One-Class SVM (OCSVM)": OCSVM(),
            "Clustering (CLUST)": CLUST(),
            "Decomposition (DECOMP)": DECOMP(),
            "Meta-model (META)": META(),
            "Variational Autoencoder (VAE)": VAE(),
            "Change Point Detection (CPD)": CPD(),
            "Mixture Models (MIXMOD)": MIXMOD(),
        }

        # Calc outliers for original data ==============================================================================
        log.info("Calc outliers for original data")
        df_outs = pd.DataFrame(index=list(classifiers.keys()), columns=list(thresholders.keys()))
        for pyod_m_name, pyod_m in classifiers.items():
            log.info(pyod_m_name)
            scores = pyod_m.fit(df.loc[:, feats].values).decision_scores_
            for pythresh_m_name, pythresh_m in thresholders.items():
                labels = pythresh_m.eval(scores)
                df_outs.at[pyod_m_name, pythresh_m_name] = sum(labels) / len(labels) * 100
                
        # Plot outliers for original data ==============================================================================
        df_outs.to_excel(f"outliers.xlsx")
        df_fig = df_outs.astype(float)
        sns.set_theme(style='ticks', font_scale=1.0)
        fig, ax = plt.subplots(figsize=(16, 12))
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

        colors_atks = {
            "MomentumIterative": px.colors.qualitative.D3[0],
            "BasicIterative": px.colors.qualitative.D3[1],
            "FastGradient": px.colors.qualitative.D3[3],
        }

        art_regressor = PyTorchRegressor(
            model=model,
            loss=model.loss_fn,
            input_shape=[len(feats)],
            optimizer=torch.optim.Adam(
                params=model.parameters(),
                lr=model.hparams.optimizer_lr,
                weight_decay=model.hparams.optimizer_weight_decay
            ),
            use_amp=False,
            opt_level="O1",
            loss_scale="dynamic",
            channels_first=True,
            clip_values=None,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0.0, 1.0),
            device_type="cpu",
        )

        attacks_names = ['MomentumIterative', 'BasicIterative', 'FastGradient']
        epsilons = sorted(list(set.union(
            set(np.linspace(0.1, 1.0, 10)),
            set(np.linspace(0.01, 0.1, 10)),
        )))
        df_eps = pd.DataFrame(index=epsilons)

        for eps_raw in epsilons:
            eps = np.array([eps_raw * iqr(df.loc[:, feat].values) for feat in feats])
            eps_step = np.array([0.2 * eps_raw * iqr(df.loc[:, feat].values) for feat in feats])

            attacks = {
                'MomentumIterative': MomentumIterativeMethod(
                    estimator=art_regressor,
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
                    estimator=art_regressor,
                    eps=eps,
                    eps_step=eps_step,
                    max_iter=100,
                    targeted=False,
                    batch_size=512,
                    verbose=True
                ),
                'FastGradient': FastGradientMethod(
                    estimator=art_regressor,
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

            for attack_name, attack in attacks.items():
                path_curr = f"Evasion/{attack_name}/eps_{eps_raw:0.4f}"
                Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

                X_atk = attack.generate(np.float32(df.loc[:, feats].values))

                df_atk = df.loc[:, [target]].copy()
                df_atk.loc[:, feats] = X_atk

                df_atk["Prediction"] = model(torch.from_numpy(np.float32(df_atk.loc[:, feats].values))).cpu().detach().numpy().ravel()
                df_atk["Error"] = df_atk["Prediction"] - df_atk[target]
                df_atk["abs(Error)"] = df_atk["Error"].abs()
                df_atk['Data'] = f'{attack_name} Eps: {eps_raw:0.4f}'
                df_atk.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')
                
                metrics = get_reg_metrics()
                metrics_cols = [f"{m}_{p}" for m in metrics for p in ids_dict]
                df_metrics = pd.DataFrame(index=metrics_cols)
                for p, ids_part in ids_dict.items():
                    for m in metrics:
                        m_val = float(metrics[m][0](torch.from_numpy(np.float32(df.loc[ids_part, "Prediction"].values)), torch.from_numpy(np.float32(df.loc[ids_part, target].values))).numpy())
                        df_metrics.at[f"{m}_{p}", 'Origin'] = m_val
                        metrics[m][0].reset()
                        m_val = float(metrics[m][0](torch.from_numpy(np.float32(df_atk.loc[ids_part, "Prediction"].values)), torch.from_numpy(np.float32(df.loc[ids_part, target].values))).numpy())
                        df_metrics.at[f"{m}_{p}", 'Attack'] = m_val
                        metrics[m][0].reset()
                df_metrics.to_excel(f"{path_curr}/metrics.xlsx", index_label='Metrics')
                
                for p in ids_dict:
                    if attack_name == 'MomentumIterative':
                        df_eps.loc[eps_raw, f"Origin_MAE_{p}"] = df_metrics.at[f'mean_absolute_error_{p}', 'Origin']
                    df_eps.loc[eps_raw, f"{attack_name}_MAE_{p}"] = df_metrics.at[f'mean_absolute_error_{p}', 'Attack']
                    
        df_eps.to_excel(f"Evasion/df_eps.xlsx", index_label='eps')

        # Plot attacks' metrics from eps ===================================================================================
        for p in ids_dict:
            df_fig = df_eps.loc[:, [f"{x}_MAE_{p}" for x in colors_atks]].copy()
            df_fig.rename(columns={f"{x}_MAE_{p}": x for x in colors_atks}, inplace=True)
            df_fig['Eps'] = df_fig.index.values
            df_fig = df_fig.melt(id_vars="Eps", var_name='Method', value_name="MAE")
            sns.set_theme(style='ticks', font_scale=1)
            fig = plt.figure()
            lines = sns.lineplot(
                data=df_fig,
                x='Eps',
                y="MAE",
                hue=f"Method",
                style=f"Method",
                palette=colors_atks,
                hue_order=list(colors_atks.keys()),
                markers=True,
                dashes=False,
            )
            plt.xscale('log')
            lines.set_xlabel(r'$\epsilon$')
            x_min = 0.009
            x_max = 1.05
            mae_basic = df_eps.at[0.01, f"Origin_MAE_{p}"]
            lines.set_xlim(x_min, x_max)
            plt.gca().plot(
                [x_min, x_max],
                [mae_basic, mae_basic],
                color='k',
                linestyle='dashed',
                linewidth=1
            )
            plt.savefig(f"Evasion/line_mae_vs_eps_{p}.png", bbox_inches='tight', dpi=200)
            plt.savefig(f"Evasion/line_mae_vs_eps_{p}.pdf", bbox_inches='tight')
            plt.close(fig)
    
    # Outliers for attacks =============================================================================================
    if config.outliers:
        epsilons_hglt = [0.05, 0.1, 0.5, 1.0]
        for atk in colors_atks:
            for eps in epsilons_hglt:
                path_curr = f"Evasion/{atk}/eps_{eps:0.4f}"
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
                fig, ax = plt.subplots(figsize=(16, 12))
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
        df_ori = df[feats].copy()
        df_ori['Class'] = 'Original'

        for atk in colors_atks:
            
            df_def_acc = pd.DataFrame(index=epsilons, columns=['Model'] + list(epsilons))
            
            for eps in tqdm(epsilons):
                
                path_curr = f"Evasion/{atk}/eps_{eps:0.4f}"
                df_adv = pd.read_excel(f"{path_curr}/df.xlsx", index_col='sample_id')
                df_adv = df_adv[feats]
                df_adv['Class'] = 'Attack'
                df_def_trn_val = pd.concat([df_ori.loc[ids_dict['trn_val'], :], df_adv.loc[ids_dict['trn_val'], :]])
                df_def_tst = pd.concat([df_ori.loc[ids_dict['tst'], :], df_adv.loc[ids_dict['tst'], :]])
                
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

                sweep_df, best_model = model_sweep(
                    task="classification",
                    train=df_def_trn_val,
                    test=df_def_tst,
                    data_config=data_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config,
                    model_list="standard",
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
                df_def_acc.at[eps, 'Model'] = best_model.config['_model_name']
                
                for tst_eps in epsilons:
                    if tst_eps != eps:
                        path_tst = f"Evasion/{atk}/eps_{tst_eps:0.4f}"
                        df_adv_tst = pd.read_excel(f"{path_tst}/df.xlsx", index_col='sample_id')
                        df_adv_tst = df_adv_tst[feats]
                        df_adv_tst['Class'] = 'Attack'
                        df_def_tst_eps = pd.concat([df_ori, df_adv_tst])
                        metrics = best_model.evaluate(test=df_def_tst_eps, verbose=False)[0]
                        df_def_acc.at[eps, tst_eps] = metrics['test_accuracy']
            df_def_acc.to_excel(f"Evasion/{atk}/detectors_accuracy.xlsx")
            
        for atk in colors_atks:
            df_def_acc = pd.read_excel(f"Evasion/{atk}/detectors_accuracy.xlsx", index_col=0)
            df_def_acc['Eps'] = [f"{x:.2f}" for x in df_def_acc.index.values]
            df_def_acc['index'] = df_def_acc['Model'] + '\n' + df_def_acc['Eps']
            df_def_acc.set_index('index', inplace=True)
            df_def_acc.drop(['Model', 'Eps'], axis=1, inplace=True)
            df_def_acc.rename(columns={x: f"{x:.2f}" for x in df_def_acc.columns}, inplace=True)
            
            df_fig = df_def_acc.astype(float)
            sns.set_theme(style='ticks', font_scale=1.0)
            fig, ax = plt.subplots(figsize=(13, 12))
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
                annot_kws={"size": 12},
                ax=ax
            )
            ax.set_xlabel('Test Attack Strength')
            ax.set_ylabel('Training Model and Data')
            heatmap_pos = heatmap.get_position()
            ax.figure.axes[-1].set_title("Accuracy")
            ax.figure.axes[-1].tick_params()
            for spine in ax.figure.axes[-1].spines.values():
                spine.set_linewidth(1)
            plt.savefig(f"Evasion/{atk}/detectors_accuracy.png", bbox_inches='tight', dpi=200)
            plt.savefig(f"Evasion/{atk}/detectors_accuracy.pdf", bbox_inches='tight')
            plt.close(fig)
