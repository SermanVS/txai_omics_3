import numpy as np
import plotly.io as pio
pio.kaleido.scope.mathjax = None
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import missingno as msno
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from txai_omics_3.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics
import pandas as pd
import torch
import patchworklib as pw


def add_iqr_outs_to_df(df, df_train, feats):
    out_columns = []
    for f in tqdm(feats):
        q1 = df_train[f].quantile(0.25)
        q3 = df_train[f].quantile(0.75)
        iqr = q3 - q1
        df[f"{f}_out_iqr"] = True
        out_columns.append(f"{f}_out_iqr")
        filter = (df[f] >= q1 - 1.5 * iqr) & (df[f] <= q3 + 1.5 * iqr)
        df.loc[filter, f"{f}_out_iqr"] = False
    df[f"n_outs_iqr"] = df.loc[:, out_columns].sum(axis=1)


def plot_iqr_outs(df, feats, color, title, path, is_msno_plots=True):
    # Plot hist for Number of IQR outliers
    hist_bins = np.linspace(-0.5, len(feats) + 0.5, len(feats) + 2)
    fig = plt.figure(figsize=(4, 3))
    sns.set_theme(style='whitegrid')
    histplot = sns.histplot(
        data=df,
        x=f"n_outs_iqr",
        multiple="stack",
        bins=hist_bins,
        edgecolor='k',
        linewidth=0.01,
        color=color,
    )
    histplot.set(xlim=(-0.5, len(feats) + 0.5))
    histplot.set_title(title)
    histplot.set_xlabel("Number of IQR outliers")
    plt.savefig(f"{path}/hist_nOuts.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/hist_nOuts.pdf", bbox_inches='tight')
    plt.close(fig)

    if is_msno_plots:
        # Prepare dataframe for msno lib
        out_columns = [f"{f}_out_iqr" for f in feats]
        df_msno = df.loc[:, out_columns].copy()
        df_msno.replace({True: np.nan}, inplace=True)
        df_msno.rename(columns=dict(zip(out_columns, feats)), inplace=True)

        # Plot barplot for features with outliers
        msno_bar = msno.bar(
            df=df_msno,
            label_rotation=90,
            color=color,
            figsize=(0.4 * len(feats), 4)
        )
        plt.xticks(ha='center')
        plt.setp(msno_bar.xaxis.get_majorticklabels(), ha="center")
        msno_bar.set_title(title, fontdict={'fontsize': 22})
        msno_bar.set_ylabel("Non-outlier samples", fontdict={'fontsize': 22})
        plt.savefig(f"{path}/bar_feats_nOuts.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"{path}/bar_feats_nOuts.pdf", bbox_inches='tight')
        plt.clf()

        # Plot matrix of samples outliers distribution
        msno_mtx = msno.matrix(
            df=df_msno,
            label_rotation=90,
            color=colors.to_rgb(color),
            figsize=(0.7 * len(feats), 5)
        )
        plt.xticks(ha='center')
        plt.setp(msno_bar.xaxis.get_majorticklabels(), ha="center")
        msno_mtx.set_title(title, fontdict={'fontsize': 22})
        msno_mtx.set_ylabel("Samples", fontdict={'fontsize': 22})
        plt.savefig(f"{path}/matrix_featsOuts.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"{path}/matrix_featsOuts.pdf", bbox_inches='tight')
        plt.clf()

        # Plot heatmap of features outliers correlations
        msno_heatmap = msno.heatmap(
            df=df_msno,
            label_rotation=90,
            cmap="bwr",
            fontsize=12,
            figsize=(0.6 * len(feats), 0.6 * len(feats))
        )
        msno_heatmap.set_title(title, fontdict={'fontsize': 22})
        plt.setp(msno_heatmap.xaxis.get_majorticklabels(), ha="center")
        msno_heatmap.collections[0].colorbar.ax.tick_params(labelsize=20)
        plt.savefig(f"{path}/heatmap_featsOutsCorr.png", bbox_inches='tight', dpi=200)
        plt.savefig(f"{path}/heatmap_featsOutsCorr.pdf", bbox_inches='tight')
        plt.clf()


def plot_iqr_outs_reg(df, title, path, col_pred, col_real, col_error):
    q25, q50, q75 = np.percentile(df['n_outs_iqr'].values, [25, 50, 75])
    df_fig = df.loc[:, [col_real, col_pred, col_error, 'n_outs_iqr']].copy()
    df_fig.loc[df_fig['n_outs_iqr'] <= round(q25), 'Type'] = 'Inlier'
    df_fig.loc[df_fig['n_outs_iqr'] >= round(q75), 'Type'] = 'Outlier'
    df_fig = df_fig.loc[df_fig['Type'].isin(['Inlier', 'Outlier']), :]

    mae_dict = {
        'Inlier': mean_absolute_error(
            df_fig.loc[df_fig['Type'] == 'Inlier', col_real].values,
            df_fig.loc[df_fig['Type'] == 'Inlier', col_pred].values
        ),
        'Outlier': mean_absolute_error(
            df_fig.loc[df_fig['Type'] == 'Outlier', col_real].values,
            df_fig.loc[df_fig['Type'] == 'Outlier', col_pred].values
        ),
    }
    _, mw_pval = mannwhitneyu(
        df_fig.loc[df_fig['Type'] == 'Inlier', col_error].values,
        df_fig.loc[df_fig['Type'] == 'Outlier', col_error].values,
        alternative='two-sided'
    )
    type_description = {
        'Inlier': f'<= q25({round(q25)}) IQR Outliers',
        'Outlier': f'>= q75({round(q75)}) IQR Outliers'
    }
    samples_num = {
        'Inlier':  df_fig[df_fig['Type'] == 'Inlier'].shape[0],
        'Outlier': df_fig[df_fig['Type'] == 'Outlier'].shape[0]
    }
    rename_dict = {x: f"{x}\n{type_description[x]}\n({samples_num[x]} samples)\nMAE={mae_dict[x]:0.2f}" for x in mae_dict}
    df_fig['Type'].replace(rename_dict, inplace=True)
    fig = plt.figure(figsize=(4, 3))
    sns.set_theme(style='whitegrid')
    violin = sns.violinplot(
        data=df_fig,
        x='Type',
        y=col_error,
        palette=['dodgerblue', 'crimson'],
        scale='width',
        order=[rename_dict['Inlier'], rename_dict['Outlier']],
        saturation=0.75,
    )
    pval_formatted = [f"{mw_pval:.2e}"]
    annotator = Annotator(
        violin,
        pairs=[(rename_dict['Inlier'], rename_dict['Outlier'])],
        data=df_fig,
        x='Type',
        y=col_error,
        order=[rename_dict['Inlier'], rename_dict['Outlier']],
    )
    annotator.set_custom_annotations(pval_formatted)
    annotator.configure(loc='outside')
    annotator.annotate()
    plt.title(title, y=1.15)
    plt.savefig(f"{path}/error.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/error.pdf", bbox_inches='tight')
    plt.close(fig)


def plot_iqr_outs_cls(df, path, col_class, col_pred, col_real, cols_prob, palette):
    q25, q50, q75 = np.percentile(df['n_outs_iqr'].values, [25, 50, 75])
    df_fig = df.loc[:, [col_real, col_pred, col_class, 'n_outs_iqr'] + cols_prob].copy()
    df_fig.loc[df_fig['n_outs_iqr'] <= q25, 'Type'] = 'Inlier'
    df_fig.loc[df_fig['n_outs_iqr'] >= q75, 'Type'] = 'Outlier'
    df_fig = df_fig.loc[df_fig['Type'].isin(['Inlier', 'Outlier']), :]

    type_description = {
        'Inlier': f'<= q25({round(q25)}) IQR Outliers',
        'Outlier': f'>= q75({round(q75)}) IQR Outliers'
    }
    samples_num = {
        'Inlier':  df_fig[df_fig['Type'] == 'Inlier'].shape[0],
        'Outlier': df_fig[df_fig['Type'] == 'Outlier'].shape[0]
    }

    metrics_pred = get_cls_pred_metrics(num_classes=2)
    metrics_prob = get_cls_prob_metrics(num_classes=2)
    df_metrics = pd.DataFrame(index=list(metrics_pred.keys()) + list(metrics_prob.keys()))
    for part in ['Inlier', 'Outlier']:
        y_real = torch.from_numpy(df_fig.loc[df_fig['Type'] == part, col_real].values.astype('int32'))
        y_pred = torch.from_numpy(df_fig.loc[df_fig['Type'] == part, col_pred].values.astype('int32'))
        y_prob = torch.from_numpy(df_fig.loc[df_fig['Type'] == part, cols_prob].values)
        for m in metrics_pred:
            m_val = float(metrics_pred[m][0](y_pred, y_real).numpy())
            metrics_pred[m][0].reset()
            df_metrics.at[m, part] = m_val
        for m in metrics_prob:
            m_val = 0
            try:
                m_val = float(metrics_prob[m][0](y_prob, y_real).numpy())
            except ValueError:
                pass
            metrics_prob[m][0].reset()
            df_metrics.at[m, part] = m_val
    df_metrics.to_excel(f"{path}/df_metrics.xlsx", index_label="Metrics")
    df_metrics_fig = df_metrics.transpose(copy=True)
    df_metrics_fig['Type'] = df_metrics_fig.index

    barplots = {}
    metrics_dict = {'accuracy_weighted': 'Accuracy', 'auroc_weighted': 'AUROC'}
    for m, m_name in metrics_dict.items():
        barplots[m] = pw.Brick(figsize=(3.5, 1.0))
        sns.set_theme(style='whitegrid')
        barplot = sns.barplot(
            data=df_metrics_fig,
            y='Type',
            hue='Type',
            x=m,
            edgecolor='black',
            palette={'Inlier': 'chartreuse', 'Outlier': 'darkorange'},
            dodge=False,
            ax=barplots[m],
        )
        barplots[m].get_legend().remove()
        for container in barplots[m].containers:
            barplots[m].bar_label(container, fmt='%.3f')
        barplots[m].set_xlabel(m_name)

    rename_dict = {x: f"{x}\n{type_description[x]}\n({samples_num[x]} samples)" for x in type_description}

    countplots = {}
    kdeplots = {}
    for dt in ['Inlier', 'Outlier']:
        countplots[dt] = pw.Brick(figsize=(2, 2))
        sns.set_theme(style='whitegrid')
        countplot = sns.countplot(
            data=df_fig.loc[df_fig['Type'] == dt, :],
            x=col_class,
            edgecolor='black',
            palette=palette,
            orient='v',
            order=list(palette.keys()),
            ax=countplots[dt]
        )
        countplots[dt].bar_label(countplot.containers[0])
        countplots[dt].set_ylabel("Count")
        countplots[dt].set_title(rename_dict[dt])

        kdeplots[dt] = pw.Brick(figsize=(5, 2))
        sns.set_theme(style='whitegrid')
        kde = sns.kdeplot(
            data=df_fig.loc[df_fig['Type'] == dt, :],
            x=cols_prob[-1],
            hue=col_class,
            linewidth=2,
            palette=palette,
            common_norm=False,
            hue_order=list(palette.keys()),
            fill=True,
            cut=0,
            ax=kdeplots[dt]
        )
        sns.move_legend(kdeplots[dt], "upper center")
        kdeplots[dt].set_ylabel(rename_dict[dt], fontsize=10)


    pw_fig = ((barplots['accuracy_weighted'] | barplots['auroc_weighted']) / (countplots['Inlier'] | kdeplots['Inlier']) / (countplots['Outlier'] | kdeplots['Outlier']))
    pw_fig.savefig(f"{path}/bar_count_kde.png", bbox_inches='tight', dpi=200)
    pw_fig.savefig(f"{path}/bar_count_kde.pdf", bbox_inches='tight')
    pw.clear()

