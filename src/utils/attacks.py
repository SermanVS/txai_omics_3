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


def plot_atk_reg_in_reduced_dimension(df, df_atk, dim_red_labels, path, title):
    df['Symbol'] = 'o'
    df_atk['Symbol'] = 'X'
    df_ori_adv = pd.concat([df, df_atk])
    norm = plt.Normalize(df_ori_adv["Error"].min(), df_ori_adv["Error"].max())
    sm = plt.cm.ScalarMappable(cmap="spring", norm=norm)
    sm.set_array([])
    for m in dim_red_labels:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.set_theme(style='whitegrid')
        scatter = sns.scatterplot(
            data=df_ori_adv,
            x=dim_red_labels[m][0],
            y=dim_red_labels[m][1],
            palette='spring',
            hue='Error',
            linewidth=1,
            alpha=0.85,
            edgecolor="k",
            style=df_ori_adv.loc[:, 'Symbol'].values,
            s=40,
            ax=ax
        )
        scatter.get_legend().remove()
        fig.colorbar(sm, label="Error")
        plt.title(f'{title}', y=1.2, fontsize=14)

        legend_handles = [
            mlines.Line2D([], [], marker='o', linestyle='None', markeredgecolor='k', markerfacecolor='lightgrey', markersize=10, label='Origin'),
            mlines.Line2D([], [], marker='X', linestyle='None', markeredgewidth=0, markerfacecolor='lightgrey', markersize=10, label='Attack')
        ]
        plt.legend(handles=legend_handles, title="Samples", bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", borderaxespad=0, mode="expand", ncol=2, frameon=False)

        plt.savefig(f"{path}/{m}.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"{path}/{m}.pdf", bbox_inches='tight')
        plt.close(fig)


def plot_atk_reg_feats_dist(df, df_atk, feats, target, data_name, color, path):
    df_ori_atk = pd.concat([df, df_atk])
    pw_brick_kdes = {}
    pw_brick_scatters = {}
    for f in feats:
        pw_brick_kdes[f] = pw.Brick(figsize=(1, 0.75))
        sns.set_theme(style='whitegrid')
        kdeplot = sns.kdeplot(
            data=df_ori_atk,
            x=f,
            hue='Data',
            palette={'Origin': 'grey', data_name: color},
            hue_order=['Origin', data_name],
            fill=True,
            common_norm=False,
            ax=pw_brick_kdes[f]
        )

        pw_brick_scatters[f] = pw.Brick(figsize=(1, 0.75))
        sns.set_theme(style='whitegrid')
        scatterplot = sns.scatterplot(
            data=df_ori_atk,
            x=f,
            y=target,
            hue='Data',
            palette={'Origin': 'grey', data_name: color},
            hue_order=['Origin', data_name],
            linewidth=0.85,
            alpha=0.85,
            edgecolor="k",
            marker='o',
            s=30,
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
