import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from txai_omics_3.datamodules.tabular import TabularDataModule
from txai_omics_3.utils import utils
import matplotlib.lines as mlines
from txai_omics_3.tasks.routines import plot_cls_dim_red, calc_confidence
from txai_omics_3.tasks.metrics import get_cls_pred_metrics
import plotly.express as px
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
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
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA
from pyod.models.lunar import LUNAR

from scipy.stats import mannwhitneyu, kruskal, iqr
from statsmodels.stats.multitest import multipletests

from art.estimators.classification.pytorch import PyTorchClassifier
from art.attacks.evasion import (
    FastGradientMethod,
    BasicIterativeMethod,
    MomentumIterativeMethod,
    ZooAttack,
    CarliniL2Method,
    ElasticNet,
    NewtonFool
)

from txai_omics_3.utils.outliers.iqr import add_iqr_outs_to_df, plot_iqr_outs, plot_iqr_outs_cls
from txai_omics_3.utils.outliers.pyod import add_pyod_outs_to_df, plot_pyod_outs, plot_pyod_outs_cls
from txai_omics_3.utils.augmentation import (
    plot_aug_column_shapes,
    plot_aug_column_pair_trends_and_correlations,
    plot_aug_in_reduced_dimension,
    plot_aug_cls_feats_dist
)

log = utils.get_logger(__name__)


def adversarial_classification(config: DictConfig):

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
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)
    model = type(model).load_from_checkpoint(checkpoint_path=f"{config.path_ckpt}")
    model.eval()
    model.freeze()

    # Get model results for original data ==============================================================================
    model.produce_probabilities = True
    y_pred_prob = model(torch.from_numpy(np.float32(df.loc[:, feats].values))).cpu().detach().numpy()
    y_pred = np.argmax(y_pred_prob, 1)
    df["Prediction"] = y_pred
    for cl_name, cl_id in classes_dict.items():
        df[f"Prob {cl_name}"] = y_pred_prob[:, cl_id]

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
    Path(f"Origin/dim_red").mkdir(parents=True, exist_ok=True)
    for m, drm in dim_red_models.items():
        dim_red_res = drm.transform(df.loc[:, feats].values)
        df.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
        df.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]
        # Plot original data in reduced dimension ======================================================================
        df_fig = df.loc[:, dim_red_labels[m] + [target, 'Prediction', f"Prob {class_names[-1]}"]]
        plot_cls_dim_red(
            df=df_fig,
            col_class='Status',
            cls_names=class_names,
            col_prob=f"Prob {class_names[-1]}",
            cols_dim_red=dim_red_labels[m],
            title='Original',
            fn=f"Origin/dim_red/{m}"
        )

    # Original data: features distributions plots ======================================================================
    Path(f"Origin/feats").mkdir(parents=True, exist_ok=True)
    df_stat = pd.DataFrame(index=feats, columns=['pval', 'pval_fdr_bh'])
    for f in feats:
        vals = {}
        for cl_name, cl_id in classes_dict.items():
            vals[cl_name] = df.loc[df[target] == cl_id, f].values
        if len(vals) > 2:
            stat, pval = kruskal(*vals.values())
        else:
            stat, pval = mannwhitneyu(*vals.values(), alternative='two-sided')
        df_stat.at[f, 'pval'] = pval
    _, df_stat.loc[:, 'pval_fdr_bh'], _, _ = multipletests(df_stat.loc[:, "pval"], 0.05, method='fdr_bh')
    df_stat.sort_values(['pval_fdr_bh'], ascending=[True], inplace=True)
    df_stat[r'$ -\log_{10}(\mathrm{p-value})$'] = -np.log10(df_stat['pval_fdr_bh'].astype(float))
    df_stat.to_excel(f"Origin/feats/stat.xlsx", index_label="Features")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df_stat,
        x=r'$ -\log_{10}(\mathrm{p-value})$',
        color='darkgreen',
        linewidth=2,
        cut=0,
        fill=True,
        ax=ax
    )
    kdeplot.set_title('Features Distribution Differences')
    plt.savefig(f"Origin/feats/kde_pval.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"Origin/feats/kde_pval.pdf", bbox_inches='tight')
    plt.close(fig)
    n_top_features = 10
    top_features = list(df_stat.index[0:n_top_features])
    df_fig = df.loc[:, top_features + [target]].copy()
    df_fig[target].replace(classes_dict_rev, inplace=True)
    df_fig = df_fig.melt(id_vars=[target], value_vars=top_features, var_name='Feature', value_name='Value')
    df_fig['Feature'].replace({x: f"{x}\npval: {df_stat.at[x, 'pval_fdr_bh']:0.2e}" for x in top_features}, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 1 * n_top_features))
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df_fig,
        x='Value',
        y='Feature',
        orient='h',
        hue='Status',
        split=True,
        linewidth=1,
        cut=0,
        inner="quart",
        ax=ax
    )
    plt.savefig(f"Origin/feats/violins.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"Origin/feats/violins.pdf", bbox_inches='tight')
    plt.close(fig)

    # Outliers methods =================================================================================================
    pyod_methods = {
        'ECOD': ECOD(contamination=config.pyod_cont),
        'LUNAR': LUNAR(),
        'LODA': LODA(contamination=config.pyod_cont),
        'INNE': INNE(contamination=config.pyod_cont),
        'IForest': IForest(contamination=config.pyod_cont),
        'KNN': KNN(contamination=config.pyod_cont),
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
    if config.outliers:
        add_iqr_outs_to_df(df, df.loc[ids_dict['trn_val'], :], feats)
        add_pyod_outs_to_df(df, pyod_methods, feats)
        Path(f"Origin/outliers_iqr").mkdir(parents=True, exist_ok=True)
        Path(f"Origin/outliers_pyod").mkdir(parents=True, exist_ok=True)

        df_fig = df.loc[:, prob_cols + [target, 'Prediction', 'n_outs_iqr', 'Detections']].copy()
        df_fig[f"{target} ID"] = df_fig[target]
        df_fig[target].replace(classes_dict_rev, inplace=True)

        plot_iqr_outs(df, feats, 'grey', 'Origin', f"Origin/outliers_iqr", is_msno_plots=False)
        plot_iqr_outs_cls(
            df=df_fig,
            path=f"Origin/outliers_iqr",
            col_class=target,
            col_pred='Prediction',
            col_real=f"{target} ID",
            cols_prob=prob_cols,
            palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
        )

        plot_pyod_outs(df, list(pyod_methods.keys()), 'grey', 'Origin', f"Origin/outliers_pyod", n_cols=4)
        plot_pyod_outs_cls(
            df=df_fig,
            path=f"Origin/outliers_pyod",
            col_class=target,
            col_pred="Prediction",
            col_real=f"{target} ID",
            cols_prob=prob_cols,
            palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
        )

    # Calc metrics' confidence intervals ===============================================================================
    Path(f"Origin/confidence").mkdir(parents=True, exist_ok=True)
    metrics = get_cls_pred_metrics(len(class_names))
    calc_confidence(df, target, 'Prediction', metrics, f"Origin/confidence", task="classification")

    # Save original data ===============================================================================================
    df.to_excel("Origin/df.xlsx", index=True)

    # Augmentation =====================================================================================================
    if config.augmentations:
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
        metadata.update_column(target, sdtype='categorical')

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
            df_col_shapes['Index'] = df_col_shapes['Column']
            df_col_shapes.set_index('Index', inplace=True)
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

            model.produce_probabilities = True
            y_pred_prob = model(torch.from_numpy(np.float32(df_aug.loc[:, feats].values))).cpu().detach().numpy()
            y_pred = np.argmax(y_pred_prob, 1)
            df_aug.loc[:, "Prediction"] = y_pred
            for cl_name, cl_id in classes_dict.items():
                df_aug.loc[:, f"Prob {cl_name}"] = y_pred_prob[:, cl_id]
            df_aug['Data'] = f'{s_name}'

            for m, drm in dim_red_models.items():
                dim_red_res = drm.transform(df_aug.loc[:, feats].values)
                df_aug.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
                df_aug.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]
            if config.outliers:
                add_iqr_outs_to_df(df_aug, df.loc[ids_dict['trn_val'], :], feats)
                add_pyod_outs_to_df(df_aug, pyod_methods, feats)
            df_aug.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')

            # Calc metrics' confidence intervals =======================================================================
            Path(f"{path_curr}/confidence").mkdir(parents=True, exist_ok=True)
            metrics = get_cls_pred_metrics(len(class_names))
            calc_confidence(df_aug, target, 'Prediction', metrics, f"{path_curr}/confidence", task="classification")

            # Plot augmented data in reduced dimension =================================================================
            Path(f"{path_curr}/dim_red").mkdir(parents=True, exist_ok=True)
            plot_aug_in_reduced_dimension(
                df,
                df_aug,
                dim_red_labels,
                f"{path_curr}/dim_red",
                s_name,
                cont_col=f"Prob {class_names[-1]}"
            )

            # Plot augmented data features distributions ===============================================================
            plot_aug_cls_feats_dist(
                df=df,
                df_aug=df_aug,
                synt_name=s_name,
                color=colors_augs[s_name],
                target=target,
                classes_dict=classes_dict,
                path=f"{path_curr}",
                df_stat=df_col_shapes,
            )

            # Plot outliers results ====================================================================================
            if config.outliers:
                Path(f"{path_curr}/outliers_iqr").mkdir(parents=True, exist_ok=True)
                Path(f"{path_curr}/outliers_pyod").mkdir(parents=True, exist_ok=True)

                df_fig = df_aug.loc[:, prob_cols + [target, 'Prediction', 'n_outs_iqr', 'Detections']].copy()
                df_fig[f"{target} ID"] = df_fig[target]
                df_fig[target].replace(classes_dict_rev, inplace=True)

                plot_iqr_outs(
                    df,
                    feats,
                    colors_augs[s_name],
                    s_name,
                    f"{path_curr}/outliers_iqr",
                    is_msno_plots=False
                )
                plot_iqr_outs_cls(
                    df=df_fig,
                    path=f"{path_curr}/outliers_iqr",
                    col_class=target,
                    col_pred='Prediction',
                    col_real=f"{target} ID",
                    cols_prob=prob_cols,
                    palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
                )

                plot_pyod_outs(
                    df,
                    list(pyod_methods.keys()),
                    colors_augs[s_name],
                    s_name,
                    f"{path_curr}/outliers_pyod",
                    n_cols=4
                )
                plot_pyod_outs_cls(
                    df=df_fig,
                    path=f"{path_curr}/outliers_pyod",
                    col_class=target,
                    col_pred="Prediction",
                    col_real=f"{target} ID",
                    cols_prob=prob_cols,
                    palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
                )

        # Confidence intervals in comparison with real data ============================================================
        if config.confidence:
            metrics_names = {
                'accuracy_weighted': 'Accuracy',
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
        df_ori['Symbol'] = 'o'
        df_ori['MarkerSize'] = 70

        for atk_type, atk_dict in atk_types.items():
            for val in atk_dict['values']:
                if atk_type == "Eps":
                    eps = np.array([val * iqr(df.loc[ids_atk, feat].values) for feat in feats])
                    eps_step = np.array([0.2 * val * iqr(df.loc[ids_atk, feat].values) for feat in feats])
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
                    df_atk.index += f'_{atk_name}_{val:0.2e}'

                    model.produce_probabilities = True
                    y_pred_prob = model(torch.from_numpy(np.float32(df_atk.loc[:, feats].values))).cpu().detach().numpy()
                    y_pred = np.argmax(y_pred_prob, 1)
                    df_atk.loc[:, "Prediction"] = y_pred
                    for cl_name, cl_id in classes_dict.items():
                        df_atk.loc[:, f"Prob {cl_name}"] = y_pred_prob[:, cl_id]
                    df_atk['Data'] = f'{atk_name} {label_suffix}'

                    for m, drm in dim_red_models.items():
                        dim_red_res = drm.transform(df_atk.loc[:, feats].values)
                        df_atk.loc[:, dim_red_labels[m][0]] = dim_red_res[:, 0]
                        df_atk.loc[:, dim_red_labels[m][1]] = dim_red_res[:, 1]
                    if config.outliers:
                        add_iqr_outs_to_df(df_atk, df.loc[ids_dict['trn_val'], :], feats)
                        add_pyod_outs_to_df(df_atk, pyod_methods, feats)
                    df_atk.to_excel(f"{path_curr}/df.xlsx", index_label='sample_id')

                    # Calc metrics' confidence intervals ===================================================================
                    Path(f"{path_curr}/confidence").mkdir(parents=True, exist_ok=True)
                    metrics = get_cls_pred_metrics(len(class_names))
                    calc_confidence(df_atk, target, 'Prediction', metrics, f"{path_curr}/confidence", task="classification")

                    Path(f"{path_curr}/dim_red").mkdir(parents=True, exist_ok=True)
                    df_atk['Symbol'] = 'X'
                    df_atk['MarkerSize'] = 50
                    df_ori_atk = pd.concat([df_ori, df_atk])
                    for m in dim_red_labels:
                        norm = plt.Normalize(
                            df_ori_atk[f"Prob {class_names[-1]}"].min(),
                            df_ori_atk[f"Prob {class_names[-1]}"].max()
                        )
                        sm = plt.cm.ScalarMappable(cmap="seismic", norm=norm)
                        sm.set_array([])
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.set_theme(style='whitegrid')
                        scatter = sns.scatterplot(
                            data=df_ori_atk,
                            x=dim_red_labels[m][0],
                            y=dim_red_labels[m][1],
                            palette='seismic',
                            hue=f"Prob {class_names[-1]}",
                            linewidth=0.5,
                            alpha=0.75,
                            edgecolor="cyan",
                            style=df_ori_atk.loc[:, 'Symbol'].values,
                            size='MarkerSize',
                            ax=ax
                        )
                        scatter.get_legend().remove()
                        scatter.figure.colorbar(sm, label=f"Prob {class_names[-1]}")
                        scatter.set_title(label_suffix, loc='left', y=1.05, fontdict={'fontsize': 20})
                        legend_handles = [
                            mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=10, label='Real'),
                            mlines.Line2D([], [], marker='X', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=7, label='Attack')
                        ]
                        plt.legend(handles=legend_handles, title="Samples", bbox_to_anchor=(0.4, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, ncol=2, frameon=False)
                        plt.savefig(f"{path_curr}/dim_red/{m}.png", bbox_inches='tight', dpi=200)
                        plt.savefig(f"{path_curr}/dim_red/{m}.pdf", bbox_inches='tight')
                        plt.close(fig)

                    if config.outliers:
                        Path(f"{path_curr}/outliers_iqr").mkdir(parents=True, exist_ok=True)
                        Path(f"{path_curr}/outliers_pyod").mkdir(parents=True, exist_ok=True)

                        df_fig = df_atk.loc[:, prob_cols + [target, 'Prediction', 'n_outs_iqr', 'Detections']].copy()
                        df_fig[f"{target} ID"] = df_fig[target]
                        df_fig[target].replace(classes_dict_rev, inplace=True)

                        plot_iqr_outs(
                            df,
                            feats,
                            atk_dict['colors'][atk_name],
                            f'{atk_name} {label_suffix}',
                            f"{path_curr}/outliers_iqr",
                            is_msno_plots=False
                        )
                        plot_iqr_outs_cls(
                            df=df_fig,
                            path=f"{path_curr}/outliers_iqr",
                            col_class=target,
                            col_pred='Prediction',
                            col_real=f"{target} ID",
                            cols_prob=prob_cols,
                            palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
                        )

                        plot_pyod_outs(
                            df,
                            list(pyod_methods.keys()),
                            atk_dict['colors'][atk_name],
                            f'{atk_name} {label_suffix}',
                            f"{path_curr}/outliers_pyod",
                            n_cols=4
                        )
                        plot_pyod_outs_cls(
                            df=df_fig,
                            path=f"{path_curr}/outliers_pyod",
                            col_class=target,
                            col_pred="Prediction",
                            col_real=f"{target} ID",
                            cols_prob=prob_cols,
                            palette={x: px.colors.qualitative.Alphabet[x_id] for x_id, x in enumerate(class_names)}
                        )

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

            if config.confidence:
                quantiles = [0.05, 0.95]
                for atk_name in atk_dict['names']:
                    df_conf = pd.DataFrame(index=atk_dict['values'], columns=[f"{m}_{q}" for m in metrics_names for q in quantiles])
                    for val in (pbar := tqdm(atk_dict['values'])):
                        pbar.set_description(f"Processing {atk_type}: {val}")
                        if atk_type == "Eps":
                            path_curr = f"Evasion/{atk_name}/eps_{val:0.4f}"
                        elif atk_type == "BSS":
                            path_curr = f"Evasion/{atk_name}/bss_{val}"
                        elif atk_type == "Eta":
                            path_curr = f"Evasion/{atk_name}/eta_{val:0.2e}"
                        else:
                            raise ValueError("Unsupported atk_type")
                        df_metrics = pd.read_excel(f"{path_curr}/confidence/metrics.xlsx", index_col='Metrics')
                        for m in metrics_names:
                            for q in quantiles:
                                df_conf.at[val, f"{m}_{q}"] = df_metrics.at[m, f"q{q}"]

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
                            linewidth=3,
                            ax=ax
                        )
                        line.get_legend().set_visible(False)
                        if atk_type in ["Eps", 'Eta']:
                            plt.xscale('log')
                        ax.set_xlabel(atk_type)
                        ax.set_ylabel(f"Confidence Intervals for {metrics_names[m]}")
                        plt.savefig(f"Evasion/{atk_name}/confidence_{m}.png", bbox_inches='tight', dpi=400)
                        plt.savefig(f"Evasion/{atk_name}/confidence_{m}.pdf", bbox_inches='tight')
                        plt.close(fig)

