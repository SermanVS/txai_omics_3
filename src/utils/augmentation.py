import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.datamodules.tabular import TabularDataModule
from src.utils import utils
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import matplotlib.lines as mlines
import patchworklib as pw
from src.tasks.routines import eval_regression
import plotly.express as px
from src.tasks.regression.shap import explain_shap
from src.models.tabular.base import get_model_framework_dict
from src.tasks.routines import plot_reg_error_dist, calc_confidence
from src.tasks.metrics import get_reg_metrics
import pickle
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
from pyod.models.lmdd import LMDD
from pyod.models.lof import LOF
from pyod.models.cof import COF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.sod import SOD
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.loda import LODA
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.lunar import LUNAR

from src.utils.outliers.iqr import add_iqr_outs_to_df, plot_iqr_outs, plot_iqr_outs_reg
from src.utils.outliers.pyod import add_pyod_outs_to_df, plot_pyod_outs, plot_pyod_outs_reg


def plot_aug_column_shapes(df_col_shapes, color, title, path):
    fig = plt.figure(figsize=(3, 5))
    sns.set_theme(style='whitegrid')
    barplot = sns.barplot(
        data=df_col_shapes,
        x="Score",
        y="Column",
        edgecolor='black',
        color=color,
        dodge=False,
        orient='h'
    )
    barplot.set_title(f"{title}")
    barplot.set_xlabel(f"KSComplement")
    barplot.set_ylabel(f"Features")
    plt.savefig(f"{path}/ColumnShapes.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/ColumnShapes.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_aug_column_pair_trends_and_correlations(feats_to_plot, df_col_pair_trends, title_pair_trends, path):
    df_corr_mtx = pd.DataFrame(
        data=np.zeros(shape=(len(feats_to_plot), len(feats_to_plot))),
        index=feats_to_plot,
        columns=feats_to_plot
    )
    df_pair_mtx = pd.DataFrame(
        index=feats_to_plot,
        columns=feats_to_plot
    )
    for index, row in df_col_pair_trends.iterrows():
        df_corr_mtx.at[row['Column 1'], row['Column 2']] = row['Real Correlation']
        df_corr_mtx.at[row['Column 2'], row['Column 1']] = row['Synthetic Correlation']
        df_pair_mtx.at[row['Column 1'], row['Column 2']] = row['Score']
        df_pair_mtx.at[row['Column 2'], row['Column 1']] = row['Score']

    fig = plt.figure()
    df_pair_mtx.fillna(value=np.nan, inplace=True)
    sns.set_theme(style='whitegrid')
    heatmap = sns.heatmap(
        data=df_pair_mtx,
        cmap='plasma',
        annot=True,
        fmt="0.2f",
        cbar_kws={'label': "Correlation Similarity"},
        mask=df_pair_mtx.isnull()
    )
    heatmap.set(xlabel="", ylabel="")
    heatmap.tick_params(axis='x', rotation=90)
    heatmap.set_title(f"{title_pair_trends}")
    plt.savefig(f"{path}/ColumnPairTrends.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/ColumnPairTrends.pdf", bbox_inches='tight')
    plt.close(fig)

    sns.set_theme(style='whitegrid')
    mtx_to_plot = df_corr_mtx.to_numpy()
    mtx_triu = np.triu(mtx_to_plot, +1)
    mtx_triu_mask = np.ma.masked_array(mtx_triu, mtx_triu == 0)
    cmap_triu = plt.get_cmap("seismic").copy()
    mtx_tril = np.tril(mtx_to_plot, -1)
    mtx_tril_mask = np.ma.masked_array(mtx_tril, mtx_tril == 0)
    cmap_tril = plt.get_cmap("PRGn").copy()
    fig, ax = plt.subplots()
    im_triu = ax.imshow(mtx_triu_mask, cmap=cmap_triu, vmin=-1, vmax=1)
    cbar_triu = ax.figure.colorbar(im_triu, ax=ax, location='right', shrink=0.7, pad=0.1)
    cbar_triu.ax.tick_params(labelsize=10)
    cbar_triu.set_label("Real Correlation", horizontalalignment='center', fontsize=12)
    im_tril = ax.imshow(mtx_tril_mask, cmap=cmap_tril, vmin=-1, vmax=1)
    cbar_tril = ax.figure.colorbar(im_tril, ax=ax, location='right', shrink=0.7, pad=0.1)
    cbar_tril.ax.tick_params(labelsize=10)
    cbar_tril.set_label("Synthetic Correlation", horizontalalignment='center', fontsize=12)
    ax.grid(None)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(df_corr_mtx.shape[1]))
    ax.set_yticks(np.arange(df_corr_mtx.shape[0]))
    ax.set_xticklabels(df_corr_mtx.columns.values)
    ax.set_yticklabels(df_corr_mtx.index.values)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    for i in range(df_corr_mtx.shape[0]):
        for j in range(df_corr_mtx.shape[1]):
            color = "black"
            if i != j:
                color = "black"
                if np.abs(mtx_tril[i, j]) > 0.5:
                    color = 'white'
                text = ax.text(j, i, f"{mtx_to_plot[i, j]:0.2f}", ha="center", va="center", color=color, fontsize=7)
    fig.tight_layout()
    plt.savefig(f"{path}/Correlations.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/Correlations.pdf", bbox_inches='tight')
    plt.clf()


def plot_aug_in_reduced_dimension(df, df_aug, dim_red_labels, path, title, cont_col='Error'):
    df_ori_aug = pd.concat([df, df_aug])
    for m in dim_red_labels:
        n_bins = 25
        x_xtd = (df_aug[dim_red_labels[m][0]].max() - df_aug[dim_red_labels[m][0]].min()) * 0.075
        x_min = df_aug[dim_red_labels[m][0]].min() - x_xtd
        x_max = df_aug[dim_red_labels[m][0]].max() + x_xtd
        x_shift = (x_max - x_min) / n_bins
        x_bin_centers = np.linspace(
            start=x_min + 0.5 * x_shift,
            stop=x_max - 0.5 * x_shift,
            num=n_bins
        )
        y_xtd = (df_aug[dim_red_labels[m][1]].max() - df_aug[dim_red_labels[m][1]].min()) * 0.075
        y_min = df_aug[dim_red_labels[m][1]].min() - y_xtd
        y_max = df_aug[dim_red_labels[m][1]].max() + y_xtd
        y_shift = (y_max - y_min) / n_bins
        y_bin_centers = np.linspace(
            start=y_min + 0.5 * y_shift,
            stop=y_max - 0.5 * y_shift,
            num=n_bins
        )
        df_heatmap_sum = pd.DataFrame(index=x_bin_centers, columns=y_bin_centers, data=np.zeros((n_bins, n_bins)))
        df_heatmap_cnt = pd.DataFrame(index=x_bin_centers, columns=y_bin_centers, data=np.zeros((n_bins, n_bins)))
        xs = df_aug.loc[:, dim_red_labels[m][0]].values
        xs_ids = np.floor((xs - x_min) / (x_shift + 1e-10)).astype(int)
        ys = df_aug.loc[:, dim_red_labels[m][1]].values
        ys_ids = np.floor((ys - y_min) / (y_shift + 1e-10)).astype(int)
        zs = df_aug.loc[:, cont_col].values
        for d_id in range(len(xs_ids)):
            df_heatmap_sum.iat[xs_ids[d_id], ys_ids[d_id]] += zs[d_id]
            df_heatmap_cnt.iat[xs_ids[d_id], ys_ids[d_id]] += 1
        df_heatmap = pd.DataFrame(
            data=df_heatmap_sum.values / df_heatmap_cnt.values,
            columns=df_heatmap_sum.columns,
            index=df_heatmap_sum.index
        )
        df_heatmap.to_excel(f"{path}/{m}_heatmap.xlsx")

        norm = plt.Normalize(df_ori_aug[cont_col].min(), df_ori_aug[cont_col].max())
        sm = plt.cm.ScalarMappable(cmap="spring", norm=norm)
        sm.set_array([])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.set_theme(style='whitegrid')

        ax.imshow(
            X=df_heatmap.transpose().iloc[::-1].values,
            extent=[x_min, x_max, y_min, y_max],
            vmin=df_ori_aug[cont_col].min(),
            vmax=df_ori_aug[cont_col].max(),
            aspect=x_shift / y_shift,
            cmap="spring",
            alpha=1.0
        )

        scatter_colors = {sample: colors.rgb2hex(sm.to_rgba(row[cont_col])) for sample, row in df.iterrows()}
        scatter = sns.scatterplot(
            data=df,
            x=dim_red_labels[m][0],
            y=dim_red_labels[m][1],
            palette=scatter_colors,
            hue=df.index,
            linewidth=1,
            alpha=0.85,
            edgecolor="k",
            marker='o',
            s=30,
            ax=ax
        )
        scatter.get_legend().remove()
        fig.colorbar(sm, label=cont_col)
        plt.title(f'{title}', y=1.2, fontsize=14)

        legend_handles = [
            mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=10, label='Real'),
            mlines.Line2D([], [], marker='s', linestyle='None', markeredgewidth=0, markerfacecolor='lightgrey', markersize=10, label='Synthetic')
        ]
        plt.legend(handles=legend_handles, title="Samples", bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, mode="expand", ncol=2, frameon=False)

        plt.savefig(f"{path}/{m}.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"{path}/{m}.pdf", bbox_inches='tight')
        plt.close(fig)


def plot_aug_reg_feats_dist(df, df_aug, feats, target, synt_name, color, path):
    df_ori_aug = pd.concat([df, df_aug])

    pw_brick_kdes = {}
    pw_brick_scatters = {}
    for f in feats:
        pw_brick_kdes[f] = pw.Brick(figsize=(1, 0.75))
        sns.set_theme(style='whitegrid')
        kdeplot = sns.kdeplot(
            data=df_ori_aug,
            x=f,
            hue='Data',
            palette={'Real': 'grey', synt_name: color},
            hue_order=['Real', synt_name],
            fill=True,
            common_norm=False,
            ax=pw_brick_kdes[f]
        )

        pw_brick_scatters[f] = pw.Brick(figsize=(1, 0.75))
        sns.set_theme(style='whitegrid')
        sns.histplot(
            data=df_aug,
            x=f,
            y=target,
            bins=30,
            discrete=(False, False),
            log_scale=(False, False),
            cbar=True,
            color=color,
            ax=pw_brick_scatters[f],
        )
        scatterplot = sns.scatterplot(
            data=df,
            x=f,
            y=target,
            hue='Data',
            palette={'Real': 'grey', synt_name: color},
            hue_order=['Real', synt_name],
            linewidth=0.85,
            alpha=0.85,
            edgecolor="k",
            marker='o',
            s=20,
            ax=pw_brick_scatters[f]
        )

    n_cols = 3
    n_rows = int(np.ceil(len(feats) / n_cols))
    pw_rows_kdes = []
    pw_rows_scatters = []
    for r_id in range(n_rows):
        pw_cols_kdes = []
        pw_cols_scatters = []
        for c_id in range(n_cols):
            rc_id = r_id * n_cols + c_id
            if rc_id < len(feats):
                f = feats[rc_id]
                pw_cols_kdes.append(pw_brick_kdes[f])
                pw_cols_scatters.append(pw_brick_scatters[f])
            else:
                empty_fig = pw.Brick(figsize=(1, 0.75))
                empty_fig.axis('off')
                pw_cols_kdes.append(empty_fig)
                pw_cols_scatters.append(empty_fig)
        pw_rows_kdes.append(pw.stack(pw_cols_kdes, operator="|", margin=0.05))
        pw_rows_scatters.append(pw.stack(pw_cols_scatters, operator="|", margin=0.05))
    pw_fig_kde = pw.stack(pw_rows_kdes, operator="/", margin=0.05)
    pw_fig_kde.savefig(f"{path}/feats_kde.png", bbox_inches='tight', dpi=200)
    pw_fig_kde.savefig(f"{path}/feats_kde.pdf", bbox_inches='tight')
    pw_fig_scatter = pw.stack(pw_rows_scatters, operator="/", margin=0.05)
    pw_fig_scatter.savefig(f"{path}/feats_scatter.png", bbox_inches='tight', dpi=200)
    pw_fig_scatter.savefig(f"{path}/feats_scatter.pdf", bbox_inches='tight')

    fig = plt.figure(figsize=(6, 4))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df_ori_aug,
        x=target,
        hue='Data',
        palette={'Real': 'grey', synt_name: color},
        hue_order=['Real', synt_name],
        fill=True,
        common_norm=False,
    )
    plt.savefig(f"{path}/target_kde.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/target_kde.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_aug_cls_feats_dist(df, df_aug, synt_name, color, target, classes_dict, df_stat, path):
    df_ori_aug = pd.concat([df, df_aug])

    countplots = {}
    kdeplots = {}
    for cl_name, cl_id in classes_dict.items():
        countplots[cl_name] = pw.Brick(figsize=(3, 1))
        sns.set_theme(style='whitegrid')
        countplot = sns.countplot(
            data=df_ori_aug.loc[df_ori_aug[target] == cl_id, :],
            y='Data',
            edgecolor='black',
            palette={'Real': 'grey', synt_name: color},
            orient='h',
            order=['Real', synt_name],
            ax=countplots[cl_name]
        )
        countplots[cl_name].bar_label(countplot.containers[0])
        countplots[cl_name].set_xlabel("Count")
        countplots[cl_name].set_title(f"{cl_name} samples")

        kdeplots[cl_name] = pw.Brick(figsize=(4, 2))
        sns.set_theme(style='whitegrid')
        kde = sns.kdeplot(
            data=df_ori_aug.loc[df_ori_aug[target] == cl_id, :],
            x=f"Prob {list(classes_dict.keys())[-1]}",
            hue='Data',
            linewidth=2,
            palette={'Real': 'grey', synt_name: color},
            hue_order=['Real', synt_name],
            fill=True,
            common_norm=False,
            cut=0,
            ax=kdeplots[cl_name]
        )
        sns.move_legend(kdeplots[cl_name], "upper center")
        kdeplots[cl_name].set_title(f"{cl_name} samples")

    df_stat = df_stat.loc[df_stat['Metric'] == "KSComplement", :].copy()
    df_stat.rename(columns={'Score': "KSComplement"}, inplace=True)

    brick_scores = pw.Brick(figsize=(7.5, 1.75))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df_stat,
        x='KSComplement',
        color='darkgreen',
        linewidth=2,
        cut=0,
        fill=True,
        ax=brick_scores
    )
    brick_scores.set_title('Features Distribution Differences')

    n_features = 5
    feats_dict = {
        'Top Features': list(df_stat.index[0:n_features]),
        'Bottom Features': list(df_stat.index[-n_features - 1:-1][::-1])
    }
    brick_feats_violins = {}
    for feats_set in feats_dict:
        df_fig = df_ori_aug.loc[:, feats_dict[feats_set] + ['Data']].copy()
        df_fig = df_fig.melt(
            id_vars=['Data'],
            value_vars=feats_dict[feats_set],
            var_name='Feature',
            value_name='Value')
        df_fig['Feature'].replace(
            {x: f"{x}\nScore: {df_stat.at[x, 'KSComplement']:0.2f}" for x in feats_dict[feats_set]},
            inplace=True
        )

        brick_feats_violins[feats_set] = pw.Brick(figsize=(2.5, 3))
        sns.set_theme(style='whitegrid')
        violin = sns.violinplot(
            data=df_fig,
            x='Value',
            y='Feature',
            orient='h',
            hue='Data',
            split=True,
            linewidth=1,
            palette={'Real': 'grey', synt_name: color},
            hue_order=['Real', synt_name],
            cut=0,
            inner="quart",
            ax=brick_feats_violins[feats_set]
        )
        brick_feats_violins[feats_set].set_title(feats_set)

    pw_fig_row_1 = pw.stack(list(countplots.values()), operator="|")
    pw_fig_row_2 = pw.stack(list(kdeplots.values()), operator="|")

    pw_fig = pw_fig_row_1 / pw_fig_row_2 / brick_scores / (brick_feats_violins['Top Features'] | brick_feats_violins['Bottom Features'])
    pw_fig.savefig(f"{path}/feats.png", bbox_inches='tight', dpi=200)
    pw_fig.savefig(f"{path}/feats.pdf", bbox_inches='tight')
    pw.clear()

