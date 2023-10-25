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
from statannotations.Annotator import Annotator
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import plotly.express as px
from src.tasks.routines import plot_reg_error_dist, calc_confidence
from src.tasks.metrics import get_reg_metrics
from scipy.stats import mannwhitneyu
from pathlib import Path
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata
from sdv.lite import SingleTablePreset
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from openTSNE import TSNE

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.sos import SOS
from pyod.models.sampling import Sampling
from pyod.models.gmm import GMM
from pyod.models.mcd import MCD
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.lunar import LUNAR

from art.estimators.regression.pytorch import PyTorchRegressor
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, MomentumIterativeMethod

from src.utils.outliers.iqr import add_iqr_outs_to_df, plot_iqr_outs
from src.utils.outliers.pyod import add_pyod_outs_to_df, plot_pyod_outs, plot_pyod_outs_reg
from src.utils.augmentation import (
    plot_aug_column_shapes,
    plot_aug_column_pair_trends_and_correlations,
    plot_aug_in_reduced_dimension,
    plot_aug_reg_feats_dist
)
from src.utils.attacks import (
    plot_atk_reg_in_reduced_dimension,
    plot_atk_reg_feats_dist
)


log = utils.get_logger(__name__)


def adversarial_regression(config: DictConfig):

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
    model.eval()
    model.freeze()

    # Get model results for original data ==============================================================================
    df["Prediction"] = model(torch.from_numpy(np.float32(df.loc[:, feats].values))).cpu().detach().numpy().ravel()
    df["Error"] = df["Prediction"] - df[target]
    df["abs(Error)"] = df["Error"].abs()
    df['Data'] = 'Origin'

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

    # Dimensionality reduction models ==================================================================================
    dim_red_labels = {
        'PCA': ['PC 1', 'PC 2'],
        't-SNE': ['t-SNE 1', 't-SNE 2'],
        'IsoMap': ['IsoMap 1', 'IsoMap 2'],
    }
    dim_red_models = {
        'PCA': PCA(n_components=2, whiten=False).fit(df.loc[ids_dict['trn_val'], feats].values),
        't-SNE': TSNE(n_components=2).fit(df.loc[ids_dict['trn_val'], feats].values),
        'IsoMap': Isomap(n_components=2, n_neighbors=5).fit(df.loc[ids_dict['trn_val'], feats].values),
    }

    # Add dimensionality reduction columns to original data ============================================================
    for m, drm in dim_red_models.items():
        dim_red_res = drm.transform(df.loc[:, feats].values)
        df.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
        df.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]

    # Outliers methods =================================================================================================
    pyod_methods = {
        'ECOD': ECOD(contamination=config.pyod_cont),
        'LUNAR': LUNAR(),
        'DeepSVDD': DeepSVDD(contamination=config.pyod_cont, verbose=0),
        'VAE': VAE(encoder_neurons=[32, 16, 8], decoder_neurons=[8, 16, 32], contamination=config.pyod_cont),
        'LODA': LODA(contamination=config.pyod_cont),
        'INNE': INNE(contamination=config.pyod_cont),
        'IForest': IForest(contamination=config.pyod_cont),
        'SOD': SOD(contamination=config.pyod_cont),
        'KNN': KNN(contamination=config.pyod_cont),
        'COF': COF(contamination=config.pyod_cont),
        'LOF': LOF(contamination=config.pyod_cont),
        'MCD': MCD(contamination=config.pyod_cont),
        'GMM': GMM(contamination=config.pyod_cont),
        'Sampling': Sampling(contamination=config.pyod_cont),
        'SOS': SOS(contamination=config.pyod_cont),
        'COPOD': COPOD(contamination=config.pyod_cont),
    }
    for method_name, method in (pbar := tqdm(pyod_methods.items())):
        pbar.set_description(f"Processing {method_name}")
        method.fit(df.loc[ids_dict['trn_val'], feats].values)

    # Add outliers columns to original data ============================================================================
    Path(f"Origin/outliers_iqr").mkdir(parents=True, exist_ok=True)
    add_iqr_outs_to_df(df, df.loc[ids_dict['trn_val'], :], feats)
    plot_iqr_outs(df, feats, 'grey', 'Origin', f"Origin/outliers_iqr", is_msno_plots=True)

    Path(f"Origin/outliers_pyod").mkdir(parents=True, exist_ok=True)
    add_pyod_outs_to_df(df, pyod_methods, feats)
    plot_pyod_outs(df, list(pyod_methods.keys()), 'grey', 'Origin', f"Origin/outliers_pyod")
    plot_pyod_outs_reg(df, 'Origin', f"Origin/outliers_pyod", 'Prediction', target, 'Error')

    # Plot regression error distributions ==============================================================================
    Path(f"Origin/errors").mkdir(parents=True, exist_ok=True)
    plot_reg_error_dist(df, feats, 'grey', 'Origin', f"Origin/errors", "abs(Error)")

    # Calc metrics' confidence intervals ===============================================================================
    Path(f"Origin/confidence").mkdir(parents=True, exist_ok=True)
    metrics = get_reg_metrics()
    calc_confidence(df, target, 'Prediction', metrics, f"Origin/confidence")

    # Save original data ===============================================================================================
    df.to_excel("Origin/df.xlsx", index=True)

    # Attacks ==========================================================================================================
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
    epsilons_hglt = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    colors_epsilons = {x: px.colors.qualitative.G10[x_id] for x_id, x in enumerate(['Origin'] + epsilons_hglt)}

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
            df_atk.index += f'_{attack_name}_{eps_raw:0.4f}'

            df_atk["Prediction"] = model(torch.from_numpy(np.float32(df_atk.loc[:, feats].values))).cpu().detach().numpy().ravel()
            df_atk["Error"] = df_atk["Prediction"] - df_atk[target]
            df_atk["abs(Error)"] = df_atk["Error"].abs()
            df_atk['Data'] = f'{attack_name} Eps: {eps_raw:0.4f}'

            for m, drm in dim_red_models.items():
                dim_red_res = drm.transform(df_atk.loc[:, feats].values)
                df_atk.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
                df_atk.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]
            add_iqr_outs_to_df(df_atk, df.loc[ids_dict['trn_val'], :], feats)
            add_pyod_outs_to_df(df_atk, pyod_methods, feats)
            df_atk.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')

            # Calc metrics' confidence intervals =======================================================================
            Path(f"{path_curr}/confidence").mkdir(parents=True, exist_ok=True)
            metrics = get_reg_metrics()
            calc_confidence(df_atk, target, 'Prediction', metrics, f"{path_curr}/confidence")

            if eps_raw in epsilons_hglt:
                # Plot augmented data in reduced dimension =============================================================
                Path(f"{path_curr}/dim_red").mkdir(parents=True, exist_ok=True)
                plot_atk_reg_in_reduced_dimension(
                    df,
                    df_atk,
                    dim_red_labels,
                    f"{path_curr}/dim_red",
                    fr'{attack_name} $\epsilon={eps_raw:0.2f}$'
                )

                # Plot augmented data features distributions ===========================================================
                plot_atk_reg_feats_dist(
                    df,
                    df_atk,
                    feats,
                    target,
                    f'{attack_name} Eps: {eps_raw:0.4f}',
                    colors_epsilons[eps_raw],
                    f"{path_curr}"
                )

                # Plot outliers results ================================================================================
                Path(f"{path_curr}/outliers_iqr").mkdir(parents=True, exist_ok=True)
                plot_iqr_outs(df_atk, feats, colors_epsilons[eps_raw], fr'{attack_name} $\epsilon={eps_raw:0.2f}$', f"{path_curr}/outliers_iqr", is_msno_plots=True)

                Path(f"{path_curr}/outliers_pyod").mkdir(parents=True, exist_ok=True)
                plot_pyod_outs(df_atk, list(pyod_methods.keys()), colors_epsilons[eps_raw], fr'{attack_name} $\epsilon={eps_raw:0.2f}$', f"{path_curr}/outliers_pyod", n_cols=4)
                plot_pyod_outs_reg(df_atk, fr'{attack_name} $\epsilon={eps_raw:0.2f}$', f"{path_curr}/outliers_pyod", 'Prediction', target, 'Error')

                # Plot regression error distributions ==================================================================
                Path(f"{path_curr}/errors").mkdir(parents=True, exist_ok=True)
                plot_reg_error_dist(df_atk, feats, colors_epsilons[eps_raw], fr'{attack_name} $\epsilon={eps_raw:0.2f}$', f"{path_curr}/errors", "abs(Error)")

    # Plot attacks' metrics from eps ===================================================================================
    metrics_names = {
        'mean_absolute_error': 'MAE',
        'pearson_corr_coef': 'Pearson rho'
    }
    for m in metrics_names:
        df_eps = pd.DataFrame(index=epsilons)
        for eps_raw in epsilons:
            for atk in attacks_names:
                path_curr = f"Evasion/{atk}/eps_{eps_raw:0.4f}"
                df_metrics = pd.read_excel(f"{path_curr}/confidence/metrics.xlsx", index_col=0)
                df_eps.at[eps_raw, atk] = df_metrics.at[m, "value"]
        df_eps['Eps'] = df_eps.index.values
        df_fig = df_eps.melt(id_vars="Eps", var_name='Method', value_name=metrics_names[m])
        fig = plt.figure()
        sns.set_theme(style='whitegrid', font_scale=1)
        lines = sns.lineplot(
            data=df_fig,
            x='Eps',
            y=metrics_names[m],
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
        metrics_base = pd.read_excel(f"Origin/confidence/metrics.xlsx", index_col=0).at[m, "value"]
        lines.set_xlim(x_min, x_max)
        plt.gca().plot(
            [x_min, x_max],
            [metrics_base, metrics_base],
            color='k',
            linestyle='dashed',
            linewidth=1
        )
        plt.savefig(f"Evasion/{m}_vs_eps.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"Evasion/{m}_vs_eps.pdf", bbox_inches='tight')
        plt.close(fig)

    quantiles = [0.05, 0.95]
    for atk in colors_atks:
        df_conf = pd.DataFrame(index=epsilons, columns=[f"{m}_{q}" for m in metrics_names for q in quantiles])
        for eps in (pbar := tqdm(epsilons)):
            pbar.set_description(f"Processing Eps: {eps}")
            df_metrics = pd.read_excel(f"Evasion/{atk}/eps_{eps:0.4f}/confidence/metrics.xlsx", index_col='Metrics')
            for m in metrics_names:
                for q in quantiles:
                    df_conf.at[eps, f"{m}_{q}"] = df_metrics.at[m, f"q{q}"]

        for m in metrics_names:
            df_fig = df_conf.loc[:, [f"{m}_{q}" for q in quantiles]].copy()
            df_fig['Type'] = df_fig.index
            df_fig = df_fig.melt(id_vars=['Type'], value_name=metrics_names[m])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.set_theme(style='ticks')
            scatter = sns.scatterplot(
                data=df_fig,
                y=metrics_names[m],
                x='Type',
                hue='Type',
                palette={x: colors_atks[atk] for x in epsilons},
                hue_order=epsilons,
                linewidth=0.2,
                alpha=0.95,
                edgecolor="black",
                s=16,
                ax=ax
            )
            scatter.get_legend().set_visible(False)
            line = sns.lineplot(
                data=df_fig,
                y=metrics_names[m],
                x='Type',
                hue='Type',
                palette={x: colors_atks[atk] for x in epsilons},
                hue_order=epsilons,
                linewidth=3,
                ax=ax
            )
            line.get_legend().set_visible(False)
            plt.xscale('log')
            ax.set_xlabel(r'$\epsilon$')
            ax.set_ylabel(f"Confidence Intervals for {metrics_names[m]}")
            plt.savefig(f"Evasion/{atk}/confidence_{m}.png", bbox_inches='tight', dpi=400)
            plt.savefig(f"Evasion/{atk}/confidence_{m}.pdf", bbox_inches='tight')
            plt.close(fig)

    # Augmentation =====================================================================================================
    df['Data'] = 'Real'

    colors_augs = {
        'FAST_ML': px.colors.qualitative.Light24[0],
        'GaussianCopula': px.colors.qualitative.Light24[1],
        'CTGANSynthesizer': px.colors.qualitative.Light24[2],
        'TVAESynthesizer': px.colors.qualitative.Light24[3],
        'CopulaGANSynthesizer': px.colors.qualitative.Light24[4],
    }

    aug_n_samples = config.aug_n_samples
    df_aug_input = df.loc[:, np.concatenate((feats, [target]))]
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_aug_input)

    synthesizers = {
        'FAST_ML': SingleTablePreset(metadata, name='FAST_ML'),
        'GaussianCopula': GaussianCopulaSynthesizer(metadata),
        'CTGANSynthesizer': CTGANSynthesizer(metadata),
        'TVAESynthesizer': TVAESynthesizer(metadata),
        'CopulaGANSynthesizer': CopulaGANSynthesizer(metadata),
    }

    for s_name, s in (pbar := tqdm(synthesizers.items())):
        pbar.set_description(f"Processing {s_name}")
        path_curr = f"Augmentation/{s_name}"
        Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

        s.fit(data=df_aug_input)
        s.save(filepath=f"{path_curr}/synthesizer.pkl")
        df_aug = s.sample(num_rows=aug_n_samples)
        quality_report = evaluate_quality(
            df_aug_input,
            df_aug,
            metadata
        )
        q_rep_prop = quality_report.get_properties()
        q_rep_prop.set_index('Property', inplace=True)
        df_col_shapes = quality_report.get_details(property_name='Column Shapes')
        df_col_shapes.sort_values(["Score"], ascending=[False], inplace=True)
        df_col_shapes.to_excel(f"{path_curr}/ColumnShapes.xlsx", index=False)
        df_col_pair_trends = quality_report.get_details(property_name='Column Pair Trends')
        df_col_pair_trends.to_excel(f"{path_curr}/ColumnPairTrends.xlsx", index=False)

        if len(feats) < 15:
            plot_aug_column_shapes(
                df_col_shapes,
                colors_augs[s_name],
                f"{s_name} Average Score: {q_rep_prop.at['Column Shapes', 'Score']:0.2f}",
                path_curr
            )
            plot_aug_column_pair_trends_and_correlations(
                np.concatenate((feats, ['Age'])),
                df_col_pair_trends,
                f"{s_name} Average Score: {q_rep_prop.at['Column Pair Trends', 'Score']:0.2f}",
                path_curr
            )

        df_aug["Prediction"] = model(torch.from_numpy(np.float32(df_aug.loc[:, feats].values))).cpu().detach().numpy().ravel()
        df_aug["Error"] = df_aug["Prediction"] - df_aug[target]
        df_aug["abs(Error)"] = df_aug["Error"].abs()
        df_aug['Data'] = s_name

        for m, drm in dim_red_models.items():
            dim_red_res = drm.transform(df_aug.loc[:, feats].values)
            df_aug.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
            df_aug.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]
        add_iqr_outs_to_df(df_aug, df.loc[ids_dict['trn_val'], :], feats)
        add_pyod_outs_to_df(df_aug, pyod_methods, feats)
        df_aug.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')

        # Plot augmented data in reduced dimension =====================================================================
        Path(f"{path_curr}/dim_red").mkdir(parents=True, exist_ok=True)
        plot_aug_in_reduced_dimension(
            df,
            df_aug,
            dim_red_labels,
            f"{path_curr}/dim_red",
            s_name
        )

        # Plot augmented data features distributions ===================================================================
        plot_aug_reg_feats_dist(
            df,
            df_aug,
            feats,
            target,
            s_name,
            colors_augs[s_name],
            f"{path_curr}"
        )

        # Plot outliers results ========================================================================================
        Path(f"{path_curr}/outliers_iqr").mkdir(parents=True, exist_ok=True)
        plot_iqr_outs(df_aug, feats, colors_augs[s_name], s_name, f"{path_curr}/outliers_iqr", is_msno_plots=True)

        Path(f"{path_curr}/outliers_pyod").mkdir(parents=True, exist_ok=True)
        plot_pyod_outs(df_aug, list(pyod_methods.keys()), colors_augs[s_name], s_name, f"{path_curr}/outliers_pyod")
        plot_pyod_outs_reg(df_aug, s_name, f"{path_curr}/outliers_pyod", 'Prediction', target, 'Error')

        # Plot regression error distributions ==========================================================================
        Path(f"{path_curr}/errors").mkdir(parents=True, exist_ok=True)
        plot_reg_error_dist(df_aug, feats, colors_augs[s_name], s_name, f"{path_curr}/errors", "abs(Error)")

        # Calc metrics' confidence intervals ===========================================================================
        Path(f"{path_curr}/confidence").mkdir(parents=True, exist_ok=True)
        metrics = get_reg_metrics()
        calc_confidence(df_aug, target, 'Prediction', metrics, f"{path_curr}/confidence")

    # Confidence intervals in comparison with real data ================================================================
    metrics_names = {
        'mean_absolute_error': 'MAE',
        'pearson_corr_coef': 'Pearson rho'
    }
    quantiles = [0.05, 0.95]

    df_conf = pd.DataFrame(
        index=['Real'] + list(colors_augs.keys()),
        columns=[f"{m}_{q}" for m in metrics_names for q in quantiles]
    )

    df_metrics = pd.read_excel(f"Origin/confidence/metrics.xlsx", index_col='Metrics')
    for m in metrics_names:
        for q in quantiles:
            df_conf.at["Real", f"{m}_{q}"] = df_metrics.at[m, f"q{q}"]

    for s_name in (pbar := tqdm(colors_augs)):
        pbar.set_description(f"Processing {s_name}")
        df_metrics = pd.read_excel(f"Augmentation/{s_name}/confidence/metrics.xlsx", index_col='Metrics')
        for m in metrics_names:
            for q in quantiles:
                df_conf.at[s_name, f"{m}_{q}"] = df_metrics.at[m, f"q{q}"]

    colors_dict = {'Real': 'grey'} | colors_augs
    for m in metrics_names:
        df_fig = df_conf.loc[:, [f"{m}_{q}" for q in quantiles]].copy()
        df_fig['Type'] = df_fig.index
        df_fig = df_fig.melt(id_vars=['Type'], value_name=metrics_names[m])
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.set_theme(style='ticks')
        scatter = sns.scatterplot(
            data=df_fig,
            x=metrics_names[m],
            y='Type',
            hue='Type',
            palette=colors_dict,
            hue_order=list(colors_dict.keys()),
            linewidth=0.2,
            alpha=0.95,
            edgecolor="black",
            s=16,
            ax=ax
        )
        scatter.get_legend().set_visible(False)
        line = sns.lineplot(
            data=df_fig,
            x=metrics_names[m],
            y='Type',
            hue='Type',
            palette=colors_dict,
            hue_order=list(colors_dict.keys()),
            linewidth=2,
            ax=ax
        )
        line.get_legend().set_visible(False)
        ax.set_xlabel(f"Confidence Intervals for {metrics_names[m]}")
        plt.savefig(f"Augmentation/confidence_{m}.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"Augmentation/confidence_{m}.pdf", bbox_inches='tight')
        plt.close(fig)

    # Error distribution comparison: real vs augmented =================================================================
    dfs_fig = [df.loc[:, ['Data', 'Error']].copy()]
    df_stat = pd.DataFrame(index=list(colors_augs.keys()), columns=['mw_pval'])
    mae_dict = {'Real': pd.read_excel(f"Origin/confidence/metrics.xlsx", index_col='Metrics').at['mean_absolute_error', 'value']}
    for s_name in (pbar := tqdm(colors_augs)):
        pbar.set_description(f"Processing {s_name}")
        mae_dict[s_name] = pd.read_excel(f"Augmentation/{s_name}/confidence/metrics.xlsx", index_col='Metrics').at['mean_absolute_error','value']

        df_aug = pd.read_excel(f"Augmentation/{s_name}/df.xlsx", index_col='sample_id')
        df_fig = df_aug.loc[:, ['Data', 'Error']].copy()
        df_fig.set_index(df_fig.index.astype(str).values + f'_{s_name}', inplace=True)
        dfs_fig.append(df_fig)
        _, df_stat.at[s_name, 'mw_pval'] = mannwhitneyu(df['Error'].values, df_fig['Error'].values, alternative='two-sided')
    _, df_stat.loc[:, "mw_pval_fdr_bh"], _, _ = multipletests(df_stat.loc[:, "mw_pval"], 0.05, method='fdr_bh')

    df_fig = pd.concat(dfs_fig)
    rename_dict = {x: f"{x}\nMAE={mae_dict[x]:0.2f}" for x in mae_dict}
    colors_dict_old = {'Real': 'grey'} | colors_augs
    colors_dict_new = {f"{x}\nMAE={mae_dict[x]:0.2f}": colors_dict_old[x] for x in rename_dict}
    df_fig['Data'].replace(rename_dict, inplace=True)
    fig = plt.figure(figsize=(12, 8))
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df_fig,
        x='Data',
        y='Error',
        palette=colors_dict_new,
        scale='width',
        order=list(colors_dict_new.keys()),
        saturation=0.75,
    )
    pval_formatted = [f"{df_stat.at[x, 'mw_pval_fdr_bh']:.2e}" for x in colors_augs]
    annotator = Annotator(
        violin,
        pairs=[(rename_dict['Real'], rename_dict[x]) for x in colors_augs],
        data=df_fig,
        x='Data',
        y='Error',
        order=list(colors_dict_new.keys()),
    )
    annotator.set_custom_annotations(pval_formatted)
    annotator.configure(loc='outside')
    annotator.annotate()
    plt.savefig(f"Augmentation/Errors.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"Augmentation/Errors.pdf", bbox_inches='tight')
    plt.close(fig)
