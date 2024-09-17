import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import patchworklib as pw
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from sklearn.metrics import mean_absolute_error
from txai_omics_3.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics
import pandas as pd
import torch


def add_pyod_outs_to_df(df, pyod_methods, feats):
    for method_name, method in pyod_methods.items():
        print(method_name)
        X = df.loc[:, feats].values
        df[f"{method_name}"] = method.predict(X)
        df[f"{method_name} anomaly score"] = method.decision_function(X)
        probability = method.predict_proba(X)
        df[f"{method_name} probability inlier"] = probability[:, 0]
        df[f"{method_name} probability outlier"] = probability[:, 1]
        df[f"{method_name} confidence"] = method.predict_confidence(X)
    df["Detections"] = df.loc[:, [f"{method}" for method in pyod_methods]].sum(axis=1)


def plot_pyod_outs(df, pyod_methods, color, title, path, n_cols=6):
    # Plot hist for Number of detections as outlier in different PyOD methods
    hist_bins = np.linspace(-0.5, len(pyod_methods) + 0.5, len(pyod_methods) + 2)
    fig = plt.figure()
    sns.set_theme(style='whitegrid')
    histplot = sns.histplot(
        data=df,
        x=f"Detections",
        multiple="stack",
        bins=hist_bins,
        discrete=True,
        edgecolor='k',
        linewidth=0.05,
        color=color,
    )
    histplot.set(xlim=(-0.5, len(pyod_methods) + 0.5))
    histplot.set_title(title)
    histplot.set_xlabel("Number of detections as outlier in different methods")
    plt.savefig(f"{path}/hist_nDetections.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/hist_nDetections.pdf", bbox_inches='tight')
    plt.close(fig)

    # Plot metrics distribution for each method
    metrics = {
        'anomaly score': 'AnomalyScore',
        'probability outlier': 'Probability',
        'confidence': 'Confidence'
    }
    colors_methods = {m: px.colors.qualitative.Alphabet[m_id] for m_id, m in enumerate(pyod_methods)}
    for m_name, m_title in metrics.items():
        n_rows = int(np.ceil(len(pyod_methods) / n_cols))
        pw_rows = []
        for r_id in range(n_rows):
            pw_cols = []
            for c_id in range(n_cols):
                rc_id = r_id * n_cols + c_id
                if rc_id < len(pyod_methods):
                    method = pyod_methods[rc_id]
                    brick = pw.Brick(figsize=(0.5, 0.75))
                    sns.set_theme(style='whitegrid')
                    data_fig = df[f"{method} {m_name}"].values
                    sns.violinplot(
                        data=data_fig,
                        color=colors_methods[method],
                        edgecolor='k',
                        cut=0,
                        ax=brick
                    )
                    brick.set(xticklabels=[])
                    brick.set_title(method)
                    brick.set_xlabel("")
                    brick.set_ylabel(m_title)
                    pw_cols.append(brick)
                else:
                    brick = pw.Brick(figsize=(0.5, 0.75))
                    brick.axis('off')
                    pw_cols.append(brick)
            pw_rows.append(pw.stack(pw_cols, operator="|", margin=0.05))
        pw_fig = pw.stack(pw_rows, operator="/", margin=0.05)
        pw_fig.savefig(f"{path}/methods_{m_title}.png", bbox_inches='tight', dpi=200)
        pw_fig.savefig(f"{path}/methods_{m_title}.pdf", bbox_inches='tight')
        pw.clear()


def plot_pyod_outs_reg(df, title, path, col_pred, col_real, col_error):
    # Plot Error distribution in inliers and outliers
    q25, q50, q75 = np.percentile(df['Detections'].values, [25, 50, 75])
    df_fig = df.loc[:, [col_real, col_pred, col_error, 'Detections']].copy()
    df_fig.loc[df_fig['Detections'] <= round(q25), 'Type'] = 'Inlier'
    df_fig.loc[df_fig['Detections'] >= round(q75), 'Type'] = 'Outlier'
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
        'Inlier': f'<= q25({round(q25)}) Detections',
        'Outlier': f'>= q75({round(q75)}) Detections'
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


def plot_pyod_outs_cls(df, path, col_class, col_pred, col_real, cols_prob, palette):
    q25, q50, q75 = np.percentile(df['Detections'].values, [25, 50, 75])
    df_fig = df.loc[:, [col_real, col_pred, col_class, 'Detections'] + cols_prob].copy()
    df_fig.loc[df_fig['Detections'] <= q25, 'Type'] = 'Inlier'
    df_fig.loc[df_fig['Detections'] >= q75, 'Type'] = 'Outlier'
    df_fig = df_fig.loc[df_fig['Type'].isin(['Inlier', 'Outlier']), :]

    type_description = {
        'Inlier': f'<= q25({round(q25)}) Detections',
        'Outlier': f'>= q75({round(q75)}) Detections'
    }
    samples_num = {
        'Inlier': df_fig[df_fig['Type'] == 'Inlier'].shape[0],
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

