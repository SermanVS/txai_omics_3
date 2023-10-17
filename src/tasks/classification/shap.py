import pandas as pd
import shap
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from src.utils import utils
from slugify import slugify


log = utils.get_logger(__name__)


def get_feature_importance(fi_data):
    feature_importance = fi_data['feature_importance']
    model = fi_data['model']
    X = fi_data['X']
    y_pred_prob = fi_data['y_pred_prob']
    y_pred_raw = fi_data['y_pred_raw']
    predict_func = fi_data['predict_func']
    features = fi_data['features']

    if feature_importance == 'shap_tree':
        explainer = shap.TreeExplainer(model)
        df_X = pd.DataFrame(data=X, columns=features["all"])
        df_X[features["cat"]] = df_X[features["cat"]].astype('int32')
        shap_values = explainer.shap_values(df_X)
        base_prob = list(np.mean(y_pred_prob, axis=0))

        base_prob_expl = []
        base_prob_num = []
        base_prob_den = 0
        for class_id in range(0, len(explainer.expected_value)):
            base_prob_num.append(np.exp(explainer.expected_value[class_id]))
            base_prob_den += np.exp(explainer.expected_value[class_id])
        for class_id in range(0, len(explainer.expected_value)):
            base_prob_expl.append(base_prob_num[class_id] / base_prob_den)
        log.info(f"Base probability check: {np.linalg.norm(np.array(base_prob) - np.array(base_prob_expl))}")

        # Сonvert raw SHAP values to probability SHAP values
        shap_values_prob = copy.deepcopy(shap_values)
        for class_id in range(0, len(shap_values)):
            for subject_id in range(0, y_pred_prob.shape[0]):

                # Сhecking raw SHAP values
                real_raw = y_pred_raw[subject_id, class_id]
                expl_raw = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                diff_raw = real_raw - expl_raw
                if abs(diff_raw) > 1e-5:
                    log.warning(f"Difference between raw for subject {subject_id} in class {class_id}: {abs(diff_raw)}")

                # Checking conversion to probability space
                real_prob = y_pred_prob[subject_id, class_id]
                expl_prob_num = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
                expl_prob_den = 0
                for c_id in range(0, len(explainer.expected_value)):
                    expl_prob_den += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
                expl_prob = expl_prob_num / expl_prob_den
                delta_prob = expl_prob - base_prob[class_id]
                diff_prob = real_prob - expl_prob
                if abs(diff_prob) > 1e-5:
                    log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

                # Сonvert raw SHAP values to probability SHAP values
                shap_contrib_logodd = np.sum(shap_values[class_id][subject_id])
                shap_contrib_prob = delta_prob
                coeff = shap_contrib_prob / shap_contrib_logodd
                for feature_id in range(0, X.shape[1]):
                    shap_values_prob[class_id][subject_id, feature_id] = shap_values[class_id][subject_id, feature_id] * coeff
                diff_check = shap_contrib_prob - sum(shap_values_prob[class_id][subject_id])
                if abs(diff_check) > 1e-5:
                    log.warning(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {diff_check}")

        shap_values = shap_values_prob
    elif feature_importance == "shap_kernel":
        explainer = shap.KernelExplainer(predict_func, X)
        shap_values = explainer.shap_values(X)
    elif feature_importance == "shap_sampling":
        explainer = shap.SamplingExplainer(predict_func, X)
        shap_values = explainer.shap_values(X)
    else:
        raise ValueError(f"Unsupported feature importance type: {feature_importance}")

    importance_values = np.zeros(len(features['all']))
    for cl_id in range(len(shap_values)):
        importance_values += np.mean(np.abs(shap_values[cl_id]), axis=0)

    feature_importances = pd.DataFrame.from_dict(
        {
            'feature': features['all'],
            'importance': importance_values
        }
    )

    return feature_importances


def explain_samples(config, y_real, y_pred, indexes, shap_values, base_values, X, feature_names, feature_labels, class_names, path):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    is_correct_pred = (np.array(y_real) == np.array(y_pred))
    mistakes_ids = np.where(is_correct_pred == False)[0]
    corrects_ids = np.where(is_correct_pred == True)[0]

    num_mistakes = min(len(mistakes_ids), config.num_examples)

    for m_id in mistakes_ids[0:num_mistakes]:
        log.info(f"Plotting sample with error {indexes[m_id]}")

        ind_save = indexes[m_id]
        if isinstance(ind_save, str):
            ind_save = slugify(ind_save)

        for cl_id, cl in enumerate(class_names):
            path_curr = f"{path}/mistakes/real({class_names[y_real[m_id]]})_pred({class_names[y_pred[m_id]]})/{ind_save}"
            Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[cl_id][m_id],
                    base_values=base_values[cl_id],
                    data=X[m_id],
                    feature_names=feature_labels,
                ),
                show=False,
                max_display=config.num_top_features,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/waterfall_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/waterfall_{cl}.png", bbox_inches='tight')
            plt.close()

            shap.plots.decision(
                base_value=base_values[cl_id],
                shap_values=shap_values[cl_id][m_id],
                features=X[m_id],
                feature_names=feature_labels,
                show=False,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/decision_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/decision_{cl}.png", bbox_inches='tight')
            plt.close()

            shap.plots.force(
                base_value=base_values[cl_id],
                shap_values=shap_values[cl_id][m_id],
                features=X[m_id],
                feature_names=feature_labels,
                show=False,
                matplotlib=True,
            )
            fig = plt.gcf()
            fig.savefig(f"{path_curr}/force_{cl}.pdf", bbox_inches='tight')
            fig.savefig(f"{path_curr}/force_{cl}.png", bbox_inches='tight')
            plt.close()

    correct_samples = {x: 0 for x in range(len(class_names))}
    for c_id in corrects_ids:
        if correct_samples[y_real[c_id]] < config.num_examples:
            log.info(f"Plotting correct sample {indexes[c_id]} for {y_real[c_id]}")
            ind_save = indexes[c_id]
            for cl_id, cl in enumerate(class_names):
                path_curr = f"{path}/corrects/{class_names[y_real[c_id]]}/{ind_save}"
                Path(f"{path_curr}").mkdir(parents=True, exist_ok=True)

                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[cl_id][c_id],
                        base_values=base_values[cl_id],
                        data=X[c_id],
                        feature_names=feature_labels
                    ),
                    show=False,
                    max_display=config.num_top_features
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/waterfall_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/waterfall_{cl}.png", bbox_inches='tight')
                plt.close()

                shap.plots.decision(
                    base_value=base_values[cl_id],
                    shap_values=shap_values[cl_id][c_id],
                    features=X[c_id],
                    feature_names=feature_labels,
                    show=False,
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/decision_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/decision_{cl}.png", bbox_inches='tight')
                plt.close()

                shap.plots.force(
                    base_value=base_values[cl_id],
                    shap_values=shap_values[cl_id][c_id],
                    features=X[c_id],
                    feature_names=feature_labels,
                    show=False,
                    matplotlib=True,
                )
                fig = plt.gcf()
                fig.savefig(f"{path_curr}/force_{cl}.pdf", bbox_inches='tight')
                fig.savefig(f"{path_curr}/force_{cl}.png", bbox_inches='tight')
                plt.close()

            correct_samples[y_real[c_id]] += 1


def explain_shap(config, expl_data):
    model = expl_data['model']
    model.produce_probabilities = True
    predict_func = expl_data['predict_func']
    df = expl_data['df']
    features_info = expl_data['features']
    features = features_info['all']
    features_labels = [features_info['labels'][f] for f in features_info['all']]
    class_names = expl_data['class_names']
    target = expl_data['target']

    ids_bkgrd = expl_data["ids"][config.shap_bkgrd]
    indexes_bkgrd = df.index[ids_bkgrd]
    X_bkgrd = df.loc[indexes_bkgrd, features].values
    if config.shap_explainer == 'Tree':
        explainer = shap.TreeExplainer(model)
    elif config.shap_explainer == "Kernel":
        explainer = shap.KernelExplainer(predict_func, X_bkgrd)
    elif config.shap_explainer == "Deep":
        explainer = shap.DeepExplainer(model, torch.from_numpy(X_bkgrd))
    elif config.shap_explainer == "Sampling":
        explainer = shap.SamplingExplainer(predict_func, X_bkgrd)
    else:
        raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

    for part in expl_data["ids"]:
        print(f"part: {part}")
        if expl_data["ids"][part] is not None and len(expl_data["ids"][part]) > 0:
            log.info(f"Calculating SHAP for {part}")
            Path(f"shap/{part}/global").mkdir(parents=True, exist_ok=True)

            ids = expl_data["ids"][part]
            indexes = df.index[ids]
            X = df.loc[indexes, features].values
            y_pred_prob = df.loc[indexes, [f"pred_prob_{cl_id}" for cl_id, cl in enumerate(class_names)]].values
            y_pred_raw = df.loc[indexes, [f"pred_raw_{cl_id}" for cl_id, cl in enumerate(class_names)]].values

            if config.shap_explainer == "Tree":
                df_X = pd.DataFrame(data=X, columns=features_info["all"])
                df_X[features_info["cat"]] = df_X[features_info["cat"]].astype('int32')
                shap_values = explainer.shap_values(df_X)

                base_prob = list(np.mean(y_pred_prob, axis=0))

                base_prob_expl = []
                base_prob_num = []
                base_prob_den = 0
                for class_id in range(0, len(explainer.expected_value)):
                    base_prob_num.append(np.exp(explainer.expected_value[class_id]))
                    base_prob_den += np.exp(explainer.expected_value[class_id])
                for class_id in range(0, len(explainer.expected_value)):
                    base_prob_expl.append(base_prob_num[class_id] / base_prob_den)
                log.info(f"Base probability check: {np.linalg.norm(np.array(base_prob) - np.array(base_prob_expl))}")

                # Сonvert raw SHAP values to probability SHAP values
                shap_values_prob = copy.deepcopy(shap_values)
                for class_id in range(0, len(class_names)):
                    for subject_id in range(0, y_pred_prob.shape[0]):

                        # Сhecking raw SHAP values
                        real_raw = y_pred_raw[subject_id, class_id]
                        expl_raw = explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id])
                        diff_raw = real_raw - expl_raw
                        if abs(diff_raw) > 1e-5:
                            log.warning(f"Difference between raw for subject {subject_id} in class {class_id}: {abs(diff_raw)}")

                        # Checking conversion to probability space
                        real_prob = y_pred_prob[subject_id, class_id]
                        expl_prob_num = np.exp(explainer.expected_value[class_id] + sum(shap_values[class_id][subject_id]))
                        expl_prob_den = 0
                        for c_id in range(0, len(explainer.expected_value)):
                            expl_prob_den += np.exp(explainer.expected_value[c_id] + sum(shap_values[c_id][subject_id]))
                        expl_prob = expl_prob_num / expl_prob_den
                        delta_prob = expl_prob - base_prob[class_id]
                        diff_prob = real_prob - expl_prob
                        if abs(diff_prob) > 1e-5:
                            log.warning(f"Difference between prediction for subject {subject_id} in class {class_id}: {abs(diff_prob)}")

                        # Сonvert raw SHAP values to probability SHAP values
                        shap_contrib_logodd = np.sum(shap_values[class_id][subject_id])
                        shap_contrib_prob = delta_prob
                        coeff = shap_contrib_prob / shap_contrib_logodd
                        for feature_id in range(0, X.shape[1]):
                            shap_values_prob[class_id][subject_id, feature_id] = shap_values[class_id][subject_id, feature_id] * coeff
                        diff_check = shap_contrib_prob - sum(shap_values_prob[class_id][subject_id])
                        if abs(diff_check) > 1e-5:
                            log.warning(f"Difference between SHAP contribution for subject {subject_id} in class {class_id}: {diff_check}")

                shap_values = shap_values_prob
                expected_value = base_prob
            elif config.shap_explainer in ["Kernel", "Sampling"]:
                shap_values = explainer.shap_values(X)
                expected_value = explainer.expected_value
                log.info(f"Base probability check: {np.linalg.norm(np.mean(y_pred_prob, axis=0) - np.array(expected_value))}")
            elif config.shap_explainer == "Deep":
                model.produce_probabilities = True
                shap_values = explainer.shap_values(torch.from_numpy(X))
                expected_value = explainer.expected_value
                log.info(f"Base probability check: {np.linalg.norm(np.mean(y_pred_prob, axis=0) - np.array(expected_value))}")
            else:
                raise ValueError(f"Unsupported explainer type: {config.shap_explainer}")

            if config.is_shap_save:
                for cl_id, cl in enumerate(expl_data['class_names']):
                    df_shap = pd.DataFrame(index=indexes,  columns=features, data=shap_values[cl_id])
                    df_shap.index.name = 'index'
                    df_shap.to_excel(f"shap/{part}/shap_{cl}.xlsx", index=True)
                    df_expected_value = pd.DataFrame()
                    df_expected_value["expected_value"] = np.array(expected_value)
                    df_expected_value.to_excel(f"shap/{part}/expected_value.xlsx", index=False)

            shap.summary_plot(
                shap_values=shap_values,
                features=X,
                feature_names=features_labels,
                max_display=config.num_top_features,
                class_names=class_names,
                class_inds=list(range(len(class_names))),
                show=False,
                color=plt.get_cmap("Set1")
            )
            plt.savefig(f'shap/{part}/global/bar.png', bbox_inches='tight')
            plt.savefig(f'shap/{part}/global/bar.pdf', bbox_inches='tight')
            plt.close()

            for cl_id, cl in enumerate(expl_data['class_names']):
                Path(f"shap/{part}/global/{cl}").mkdir(parents=True, exist_ok=True)
                shap.summary_plot(
                    shap_values=shap_values[cl_id],
                    features=X,
                    feature_names=features_labels,
                    max_display=config.num_top_features,
                    show=False,
                    plot_type="bar"
                )
                plt.savefig(f'shap/{part}/global/{cl}/bar.png', bbox_inches='tight')
                plt.savefig(f'shap/{part}/global/{cl}/bar.pdf', bbox_inches='tight')
                plt.close()

                shap.summary_plot(
                    shap_values=shap_values[cl_id],
                    features=X,
                    feature_names=features_labels,
                    max_display=config.num_top_features,
                    plot_type="violin",
                    show=False,
                )
                plt.savefig(f"shap/{part}/global/{cl}/beeswarm.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/global/{cl}/beeswarm.pdf", bbox_inches='tight')
                plt.close()

                explanation = shap.Explanation(
                    values=shap_values[cl_id],
                    base_values=np.array([expected_value] * len(ids)),
                    data=X,
                    feature_names=features_labels
                )
                shap.plots.heatmap(
                    explanation,
                    show=False,
                    max_display=config.num_top_features,
                    instance_order=explanation.sum(1)
                )
                plt.savefig(f"shap/{part}/global/{cl}/heatmap.png", bbox_inches='tight')
                plt.savefig(f"shap/{part}/global/{cl}/heatmap.pdf", bbox_inches='tight')
                plt.close()

                Path(f"shap/{part}/features/{cl}").mkdir(parents=True, exist_ok=True)
                shap_values_class = shap_values[cl_id]
                mean_abs_impact = np.mean(np.abs(shap_values_class), axis=0)
                features_order = np.argsort(mean_abs_impact)[::-1]
                inds_to_plot = features_order[0:config.num_examples]
                for feat_id, ind in enumerate(inds_to_plot):
                    feat = features[ind]
                    shap.dependence_plot(
                        ind=ind,
                        shap_values=shap_values_class,
                        features=X,
                        feature_names=features_labels,
                        show=False,
                    )
                    plt.savefig(f"shap/{part}/features/{cl}/{feat_id}_{feat}.png", bbox_inches='tight')
                    plt.savefig(f"shap/{part}/features/{cl}/{feat_id}_{feat}.pdf", bbox_inches='tight')
                    plt.close()

            mean_abs_shap_values = np.sum([np.mean(np.absolute(shap_values[cl_id]), axis=0) for cl_id, cl in enumerate(class_names)], axis=0)
            order = np.argsort(mean_abs_shap_values)[::-1]
            features_sorted = np.asarray(features)[order]

            explain_samples(
                config,
                df.loc[indexes, target].values,
                df.loc[indexes, "pred"].values,
                indexes,
                shap_values,
                expected_value,
                X,
                features,
                features_labels,
                class_names,
                f"shap/{part}/samples"
            )
