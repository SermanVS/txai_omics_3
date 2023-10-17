import pandas as pd
from src.tasks.metrics import get_cls_pred_metrics, get_cls_prob_metrics, get_reg_metrics
from sklearn.metrics import confusion_matrix
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
import torch
import patchworklib as pw
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator
from torchmetrics import BootStrapper


def save_feature_importance(df, num_features):
    if df is not None:
        df.sort_values(['importance'], ascending=[False], inplace=True)
        df['importance'] = df['importance'] / df['importance'].sum()
        df_fig = df.iloc[0:num_features, :]
        plt.figure(figsize=(8, 0.3 * df_fig.shape[0]))
        sns.set_theme(style='whitegrid')
        bar = sns.barplot(
            data=df_fig,
            y='feature_label',
            x='importance',
            edgecolor='black',
            orient='h',
            dodge=True
        )
        bar.set_xlabel("Importance")
        bar.set_ylabel("")
        plt.savefig(f"feature_importance.png", bbox_inches='tight', dpi=400)
        plt.savefig(f"feature_importance.pdf", bbox_inches='tight')
        plt.close()
        df.set_index('feature', inplace=True)
        df.to_excel("feature_importance.xlsx", index=True)


def eval_classification(config, class_names, y_real, y_pred, y_pred_prob, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics_pred = get_cls_pred_metrics(config.out_dim)
    metrics_prob = get_cls_prob_metrics(config.out_dim)

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics_pred:
                wandb.define_metric(f"{part}/{m}", summary=metrics_pred[m][1])
            for m in metrics_prob:
                wandb.define_metric(f"{part}/{m}", summary=metrics_prob[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics_pred] + [m for m in metrics_prob], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics_pred:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics_pred[m][0](y_pred_torch, y_real_torch).numpy())
        metrics_pred[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val
    for m in metrics_prob:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_prob_torch = torch.from_numpy(y_pred_prob)
        m_val = 0
        try:
            m_val = float(metrics_prob[m][0](y_pred_prob_torch, y_real_torch).numpy())
        except ValueError:
            pass
        metrics_prob[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=file_suffix)

    return metrics_df


def eval_regression(config, y_real, y_pred, loggers, part, is_log=True, is_save=True, file_suffix=''):
    metrics = get_reg_metrics()

    if is_log:
        if 'wandb' in config.logger:
            for m in metrics:
                wandb.define_metric(f"{part}/{m}", summary=metrics[m][1])

    metrics_df = pd.DataFrame(index=[m for m in metrics], columns=[part])
    metrics_df.index.name = 'metric'
    log_dict = {}
    for m in metrics:
        y_real_torch = torch.from_numpy(y_real)
        y_pred_torch = torch.from_numpy(y_pred)
        m_val = float(metrics[m][0](y_pred_torch, y_real_torch).numpy())
        metrics[m][0].reset()
        metrics_df.at[m, part] = m_val
        log_dict[f"{part}/{m}"] = m_val

    if loggers is not None:
        for logger in loggers:
            if is_log:
                logger.log_metrics(log_dict)

    if is_save:
        metrics_df.to_excel(f"metrics_{part}{file_suffix}.xlsx", index=True)

    return metrics_df


def plot_confusion_matrix(y_real, y_pred, class_names, part, suffix=''):
    cm = confusion_matrix(y_real, y_pred)
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
        fig, ax = plt.subplots(figsize=(2*len(class_names), 2*len(class_names)))
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)
        plt.savefig(f"confusion_matrix_{part}{suffix}.png", bbox_inches='tight')
        plt.savefig(f"confusion_matrix_{part}{suffix}.pdf", bbox_inches='tight')
        plt.close()


def eval_loss(loss_info, loggers, is_log=True, is_save=True, file_suffix=''):
    for epoch_id, epoch in enumerate(loss_info['epoch']):
        log_dict = {
            'epoch': loss_info['epoch'][epoch_id],
            'trn/loss': loss_info['trn/loss'][epoch_id],
            'val/loss': loss_info['val/loss'][epoch_id]
        }
        if loggers is not None:
            for logger in loggers:
                if is_log:
                    logger.log_metrics(log_dict)

    if is_save:
        loss_df = pd.DataFrame(loss_info)
        loss_df.set_index('epoch', inplace=True)
        loss_df.to_excel(f"loss{file_suffix}.xlsx", index=True)


def plot_reg_error_dist(df, feats, color, title, path, col_error_abs):
    q25, q50, q75 = np.percentile(df[col_error_abs].values, [25, 50, 75])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.set_theme(style='whitegrid')
    kdeplot = sns.kdeplot(
        data=df,
        x=col_error_abs,
        color=color,
        linewidth=4,
        cut=0,
        ax=ax
    )
    kdeplot.set_title(title)
    kdeline = ax.lines[0]
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    ax.vlines(q50, 0, np.interp(q50, xs, ys), color='black', ls=':')
    ax.fill_between(xs, 0, ys, where=(q25 <= xs) & (xs <= q75), facecolor=color, alpha=0.9)
    ax.fill_between(xs, 0, ys, where=(xs <= q25), interpolate=True, facecolor='dodgerblue', alpha=0.9)
    ax.fill_between(xs, 0, ys, where=(xs >= q75), interpolate=True, facecolor='crimson', alpha=0.9)
    plt.savefig(f"{path}/kde_error_abs.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{path}/kde_error_abs.pdf", bbox_inches='tight')
    plt.close(fig)

    df_fig = df.loc[(df[col_error_abs] <= q25) | (df[col_error_abs] >= q75), list(feats) + [col_error_abs]].copy()
    df_fig.loc[df[col_error_abs] <= q25, 'abs(Error)'] = '<q25'
    df_fig.loc[df[col_error_abs] >= q75, 'abs(Error)'] = '>q75'

    df_stat = pd.DataFrame(index=list(feats))
    for feat in list(feats):
        vals = {}
        for group in ['<q25', '>q75']:
            vals[group] = df_fig.loc[df_fig['abs(Error)'] == group, feat].values
            df_stat.at[feat, f"mean_{group}"] = np.mean(vals[group])
            df_stat.at[feat, f"median_{group}"] = np.median(vals[group])
            df_stat.at[feat, f"q75_{group}"], df_stat.at[feat, f"q25_{group}"] = np.percentile(vals[group], [75, 25])
            df_stat.at[feat, f"iqr_{group}"] = df_stat.at[feat, f"q75_{group}"] - df_stat.at[feat, f"q25_{group}"]
        _, df_stat.at[feat, "mw_pval"] = mannwhitneyu(vals['<q25'], vals['>q75'], alternative='two-sided')

    _, df_stat.loc[feats, "mw_pval_fdr_bh"], _, _ = multipletests(df_stat.loc[feats, "mw_pval"], 0.05, method='fdr_bh')
    df_stat.sort_values([f"mw_pval_fdr_bh"], ascending=[True], inplace=True)
    df_stat.to_excel(f"{path}/feats_stat.xlsx", index_label='Features')

    feats_sorted = df_stat.index.values
    axs = {}
    pw_rows = []
    n_cols = 5
    n_rows = int(np.ceil(len(feats_sorted) / n_cols))
    for r_id in range(n_rows):
        pw_cols = []
        for c_id in range(n_cols):
            rc_id = r_id * n_cols + c_id
            if rc_id < len(feats_sorted):
                feat = feats_sorted[rc_id]
                axs[feat] = pw.Brick(figsize=(0.5, 1))
                sns.set_theme(style='whitegrid')
                sns.violinplot(
                    data=df_fig,
                    x='abs(Error)',
                    y=feat,
                    palette={'<q25': 'dodgerblue', '>q75': 'crimson'},
                    scale='width',
                    order=['<q25', '>q75'],
                    saturation=0.75,
                    cut=0,
                    ax=axs[feat]
                )
                mw_pval = df_stat.at[feat, "mw_pval_fdr_bh"]
                pval_formatted = [f'{mw_pval:.2e}']
                annotator = Annotator(
                    axs[feat],
                    pairs=[('<q25', '>q75')],
                    data=df_fig,
                    x='abs(Error)',
                    y=feat,
                    order=['<q25', '>q75'],
                )
                annotator.set_custom_annotations(pval_formatted)
                annotator.configure(loc='outside')
                annotator.annotate()
                pw_cols.append(axs[feat])
            else:
                empty_fig = pw.Brick(figsize=(0.5, 1))
                empty_fig.axis('off')
                pw_cols.append(empty_fig)

        pw_rows.append(pw.stack(pw_cols, operator="|", margin=0.05))
    pw_fig = pw.stack(pw_rows, operator="/", margin=0.05)
    pw_fig.savefig(f"{path}/feats_violins.png", bbox_inches='tight', dpi=200)
    pw_fig.savefig(f"{path}/feats_violins.pdf", bbox_inches='tight')
    pw.clear()


def calc_confidence(df, col_real, col_pred, metrics, path):
    torch.manual_seed(42)
    quantiles = torch.tensor([0.05, 0.95])
    df_metrics = pd.DataFrame(index=list(metrics.keys()), columns=['mean', 'std', 'q0.05', 'q0.95'])
    y_real = torch.from_numpy(np.float32(df[col_real].values))
    y_pred = torch.from_numpy(np.float32(df[col_pred].values))
    for metric_name, metric_pair in metrics.items():
        metric = metric_pair[0]
        bootstrap = BootStrapper(
            metric,
            num_bootstraps=200,
            sampling_strategy="multinomial",
            quantile=quantiles
        )
        bootstrap.update(y_pred, y_real)
        bootstrap_output = bootstrap.compute()
        df_metrics.at[metric_name, 'mean'] = bootstrap_output['mean'].detach().cpu().numpy()
        df_metrics.at[metric_name, 'std'] = bootstrap_output['std'].detach().cpu().numpy()
        df_metrics.at[metric_name, 'q0.05'] = bootstrap_output['quantile'].detach().cpu().numpy()[0]
        df_metrics.at[metric_name, 'q0.95'] = bootstrap_output['quantile'].detach().cpu().numpy()[1]
        df_metrics.at[metric_name, 'value'] = float(metric_pair[0](y_pred, y_real).numpy())
    df_metrics.to_excel(f"{path}/metrics.xlsx", index_label='Metrics')
