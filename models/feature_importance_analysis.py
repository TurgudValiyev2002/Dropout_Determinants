from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from models.evaluation import classification_metrics
from models.preprocessing import TabularOneHotScaler

warnings.filterwarnings("ignore")


RANDOM_STATE = 123
PREP_DIR = Path("results") / "2.Model Processing"
STAGED_TUNED_DIR = Path("results") / "4.Hyperparameter Tuning"
FULL_TUNED_DIR = Path("results") / "5.Full Feature Final Modeling" / "Hyp Tuned"
ABLATION_TUNED_DIR = Path("results") / "6.Ablation Study" / "2.Academic + demographic-financial" / "Hyp Tuned"
OUT_DIR = Path("results") / "7.Feature Importance"
FI_DIR = OUT_DIR / "Feature Importance"
SHAP_DIR = OUT_DIR / "SHAP"
REPORT_DIR = OUT_DIR / "reports"
CSV_DIR = OUT_DIR / "csv"
PLOT_COLOR = "#59A14F"


def mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def slug(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def build_preprocessor(
    df: pd.DataFrame,
    features: list[str],
    metadata: dict,
    feature_order: str,
) -> tuple[TabularOneHotScaler, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    if feature_order == "stage_feature_order":
        numeric = [f for f in features if f in metadata["feature_types"]["numeric_features"]]
        categorical = [f for f in features if f in metadata["feature_types"]["categorical_features"]]
    else:
        numeric = [f for f in metadata["feature_types"]["numeric_features"] if f in features]
        categorical = [f for f in metadata["feature_types"]["categorical_features"] if f in features]
    split_frames = {split: df[df["Split"] == split].copy() for split in ["train", "validation", "test"]}
    preprocessor = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)
    x_train = preprocessor.fit_transform(split_frames["train"][features])
    x_frames = {"train": x_train}
    for split in ["validation", "test"]:
        x_frames[split] = preprocessor.transform(split_frames[split][features])
    x_arrays = {split: x.to_numpy(dtype=np.float32) for split, x in x_frames.items()}
    return preprocessor, split_frames, x_arrays


def encoded_to_raw_feature(encoded_feature: str, raw_features: list[str]) -> str:
    if encoded_feature in raw_features:
        return encoded_feature
    for feature in sorted(raw_features, key=len, reverse=True):
        if encoded_feature.startswith(f"{feature}_"):
            return feature
    return encoded_feature


def predict_proba_aligned(model, x, labels) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(x)
    model_classes = list(getattr(model, "classes_", labels))
    aligned = np.zeros((len(x), len(labels)), dtype=float)
    for idx, label in enumerate(labels):
        if label in model_classes:
            aligned[:, idx] = proba[:, model_classes.index(label)]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def metrics_for(model, x: np.ndarray, y: np.ndarray, labels: np.ndarray) -> dict:
    pred = np.asarray(model.predict(x)).ravel()
    proba = predict_proba_aligned(model, x, labels)
    return classification_metrics(y, pred, proba)


def permutation_importance_raw(
    model,
    preprocessor: TabularOneHotScaler,
    test_raw: pd.DataFrame,
    y_test: np.ndarray,
    features: list[str],
    labels: np.ndarray,
    repeats: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    baseline_x = preprocessor.transform_numpy(test_raw[features])
    baseline = metrics_for(model, baseline_x, y_test, labels)
    rows = []
    for feature in features:
        macro_drops = []
        bal_drops = []
        log_increases = []
        for repeat in range(repeats):
            shuffled = test_raw[features].copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            x_perm = preprocessor.transform_numpy(shuffled)
            perm = metrics_for(model, x_perm, y_test, labels)
            macro_drops.append(baseline["macro_f1"] - perm["macro_f1"])
            bal_drops.append(baseline["balanced_accuracy"] - perm["balanced_accuracy"])
            log_increases.append(perm["log_loss"] - baseline["log_loss"])
        rows.append(
            {
                "feature": feature,
                "baseline_macro_f1": baseline["macro_f1"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "baseline_log_loss": baseline["log_loss"],
                "macro_f1_drop_mean": float(np.mean(macro_drops)),
                "macro_f1_drop_std": float(np.std(macro_drops)),
                "balanced_accuracy_drop_mean": float(np.mean(bal_drops)),
                "log_loss_increase_mean": float(np.mean(log_increases)),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1_drop_mean", ascending=False).reset_index(drop=True)


def model_label_context(model, df: pd.DataFrame, target: str, class_to_id: dict[str, int]) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    model_classes = np.asarray(getattr(model, "classes_", []))
    uses_string_labels = len(model_classes) > 0 and isinstance(model_classes[0], str)
    if uses_string_labels:
        labels = model_classes
        y_by_split = {
            split: df[df["Split"] == split][target].to_numpy()
            for split in ["train", "validation", "test"]
        }
        dropout_idx = list(labels).index("Dropout")
    else:
        labels = model_classes if len(model_classes) else np.array(list(class_to_id.values()))
        y_by_split = {
            split: df[df["Split"] == split][target].map(class_to_id).to_numpy(dtype=int)
            for split in ["train", "validation", "test"]
        }
        dropout_idx = list(labels).index(class_to_id["Dropout"])
    return y_by_split, labels, dropout_idx


def plot_top_bar(df: pd.DataFrame, value_col: str, out_path: Path, title: str, top_n: int = 10) -> None:
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(top["feature"], top[value_col], color=PLOT_COLOR)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def get_shap_values(model, x: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        return np.stack(values, axis=2)
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, :, None]
    return values


def shap_tables(
    shap_values: np.ndarray,
    encoded_features: list[str],
    raw_features: list[str],
    dropout_idx: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = pd.DataFrame(
        {
            "encoded_feature": encoded_features,
            "feature": [encoded_to_raw_feature(col, raw_features) for col in encoded_features],
        }
    )
    global_encoded = np.mean(np.abs(shap_values), axis=(0, 2))
    dropout_encoded = np.mean(np.abs(shap_values[:, :, dropout_idx]), axis=0)
    mapping["global_mean_abs_shap_encoded"] = global_encoded
    mapping["dropout_mean_abs_shap_encoded"] = dropout_encoded

    raw = (
        mapping.groupby("feature", as_index=False)
        .agg(
            global_mean_abs_shap=("global_mean_abs_shap_encoded", "sum"),
            dropout_mean_abs_shap=("dropout_mean_abs_shap_encoded", "sum"),
            encoded_columns=("encoded_feature", "count"),
        )
        .sort_values("global_mean_abs_shap", ascending=False)
    )
    dropout = raw.sort_values("dropout_mean_abs_shap", ascending=False).reset_index(drop=True)
    return raw.reset_index(drop=True), dropout


def selected_experiments(metadata: dict) -> list[dict]:
    staged_best = pd.read_csv(STAGED_TUNED_DIR / "csv" / "best_tuned_validation_model_by_stage.csv")
    full_best = pd.read_csv(FULL_TUNED_DIR / "csv" / "full_feature_tuned_validation_model_ranking.csv").iloc[0]
    ablation_best = pd.read_csv(ABLATION_TUNED_DIR / "csv" / "ablation_tuned_validation_model_ranking.csv").iloc[0]

    experiments = []
    for _, row in staged_best.iterrows():
        stage = row["stage"]
        experiments.append(
            {
                "experiment_key": stage,
                "display_name": {
                    "A_pre_enrollment": "Stage A tuned",
                    "B_first_semester": "Stage B tuned",
                    "C_second_semester": "Stage C tuned",
                }.get(stage, stage),
                "model": row["model"],
                "model_key": row["model_key"],
                "model_path": Path(row["model_path"]),
                "features": metadata["stage_feature_sets"][stage],
                "feature_order": "stage_feature_order",
            }
        )
    experiments.append(
        {
            "experiment_key": "full_feature",
            "display_name": "Full feature tuned",
            "model": full_best["model"],
            "model_key": full_best["model_key"],
            "model_path": Path(full_best["model_path"]),
            "features": metadata["ablation_feature_sets"]["full"],
            "feature_order": "metadata_order",
        }
    )
    experiments.append(
        {
            "experiment_key": "academic_plus_demographic_financial",
            "display_name": "Academic + demographic-financial tuned",
            "model": ablation_best["model"],
            "model_key": ablation_best["model_key"],
            "model_path": Path(ablation_best["model_path"]),
            "features": metadata["ablation_feature_sets"]["academic_plus_financial_demographic"],
            "feature_order": "metadata_order",
        }
    )
    return experiments


def main() -> None:
    mkdirs(FI_DIR, SHAP_DIR, REPORT_DIR, CSV_DIR)
    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")
    target = metadata["targets"]["multiclass"]
    class_names = sorted(df[target].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    experiment_rows = []
    all_perm = []
    all_shap_global = []
    all_shap_dropout = []
    top_feature_sets = {}

    for exp in selected_experiments(metadata):
        exp_slug = slug(exp["experiment_key"])
        fi_exp_dir = FI_DIR / exp_slug
        shap_exp_dir = SHAP_DIR / exp_slug
        mkdirs(fi_exp_dir, shap_exp_dir)

        with open(exp["model_path"], "rb") as f:
            model = pickle.load(f)
        y_by_split, labels, dropout_idx = model_label_context(model, df, target, class_to_id)
        preprocessor, split_frames, x_arrays = build_preprocessor(df, exp["features"], metadata, exp["feature_order"])
        test_metrics = metrics_for(model, x_arrays["test"], y_by_split["test"], labels)

        perm = permutation_importance_raw(
            model=model,
            preprocessor=preprocessor,
            test_raw=split_frames["test"],
            y_test=y_by_split["test"],
            features=exp["features"],
            labels=labels,
            repeats=10,
        )
        perm.insert(0, "experiment", exp["display_name"])
        perm.insert(1, "experiment_key", exp["experiment_key"])
        perm.to_csv(fi_exp_dir / "permutation_importance_all_features.csv", index=False)
        perm.head(10).to_csv(fi_exp_dir / "permutation_importance_top10.csv", index=False)
        plot_top_bar(
            perm,
            "macro_f1_drop_mean",
            fi_exp_dir / "permutation_importance_top10_macro_f1.png",
            f"{exp['display_name']}: Permutation Importance",
        )
        all_perm.append(perm)

        x_test_df = pd.DataFrame(x_arrays["test"], columns=preprocessor.output_columns_)
        shap_values = get_shap_values(model, x_test_df)
        global_raw, dropout_raw = shap_tables(
            shap_values=shap_values,
            encoded_features=preprocessor.output_columns_,
            raw_features=exp["features"],
            dropout_idx=dropout_idx,
        )
        global_raw.insert(0, "experiment", exp["display_name"])
        global_raw.insert(1, "experiment_key", exp["experiment_key"])
        dropout_raw.insert(0, "experiment", exp["display_name"])
        dropout_raw.insert(1, "experiment_key", exp["experiment_key"])
        global_raw.to_csv(shap_exp_dir / "global_shap_importance_all_features.csv", index=False)
        global_raw.head(10).to_csv(shap_exp_dir / "global_shap_importance_top10.csv", index=False)
        dropout_raw.to_csv(shap_exp_dir / "dropout_class_shap_importance_all_features.csv", index=False)
        dropout_raw.head(10).to_csv(shap_exp_dir / "dropout_class_shap_importance_top10.csv", index=False)
        plot_top_bar(
            global_raw.rename(columns={"global_mean_abs_shap": "value"}).assign(feature=global_raw["feature"]),
            "value",
            shap_exp_dir / "global_shap_importance_top10.png",
            f"{exp['display_name']}: Global SHAP",
        )
        plot_top_bar(
            dropout_raw.rename(columns={"dropout_mean_abs_shap": "value"}).assign(feature=dropout_raw["feature"]),
            "value",
            shap_exp_dir / "dropout_class_shap_importance_top10.png",
            f"{exp['display_name']}: Dropout-Class SHAP",
        )
        all_shap_global.append(global_raw)
        all_shap_dropout.append(dropout_raw)
        top_feature_sets[exp["display_name"]] = set(dropout_raw.head(10)["feature"])

        experiment_rows.append(
            {
                "experiment": exp["display_name"],
                "experiment_key": exp["experiment_key"],
                "model": exp["model"],
                "model_key": exp["model_key"],
                "raw_features": len(exp["features"]),
                "encoded_features": len(preprocessor.output_columns_),
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_log_loss": test_metrics["log_loss"],
                "top_permutation_feature": perm.iloc[0]["feature"],
                "top_global_shap_feature": global_raw.iloc[0]["feature"],
                "top_dropout_shap_feature": dropout_raw.iloc[0]["feature"],
            }
        )

    experiments_df = pd.DataFrame(experiment_rows)
    experiments_df.to_csv(CSV_DIR / "feature_importance_experiment_summary.csv", index=False)
    pd.concat(all_perm, ignore_index=True).to_csv(CSV_DIR / "all_permutation_importance.csv", index=False)
    pd.concat(all_shap_global, ignore_index=True).to_csv(CSV_DIR / "all_global_shap_importance.csv", index=False)
    pd.concat(all_shap_dropout, ignore_index=True).to_csv(CSV_DIR / "all_dropout_class_shap_importance.csv", index=False)

    overlap_rows = []
    names = list(top_feature_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = sorted(top_feature_sets[left] & top_feature_sets[right])
            union = sorted(top_feature_sets[left] | top_feature_sets[right])
            overlap_rows.append(
                {
                    "left_experiment": left,
                    "right_experiment": right,
                    "overlap_count_top10_dropout_shap": len(inter),
                    "jaccard_top10_dropout_shap": len(inter) / len(union) if union else 0,
                    "overlapping_features": "; ".join(inter),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(CSV_DIR / "dropout_shap_top10_overlap_across_models.csv", index=False)

    key_models = ["Stage C tuned", "Full feature tuned", "Academic + demographic-financial tuned"]
    key_overlap = overlap_df[
        overlap_df["left_experiment"].isin(key_models) & overlap_df["right_experiment"].isin(key_models)
    ]
    key_overlap.to_csv(CSV_DIR / "key_model_dropout_shap_top10_overlap.csv", index=False)

    # Compact report.
    summary_sorted = experiments_df.sort_values("test_macro_f1", ascending=False)
    report_lines = [
        "# Feature Importance and SHAP Interpretation Report",
        "",
        "This stage explains the tuned models selected by validation performance: Stage A, Stage B, Stage C, the full-feature model, and the academic + demographic/financial ablation model. Feature importance is reported in two complementary ways: permutation importance on the held-out test set and SHAP importance for the tuned LightGBM models.",
        "",
        "Top-10 features are recommended for the paper instead of top-20 because they are easier to read and more defensible in a limited manuscript table or figure. Full feature tables are still saved for auditability.",
        "",
        "## Model-Level Summary",
        "",
    ]
    for _, row in summary_sorted.iterrows():
        report_lines.append(
            f"- {row['experiment']} ({row['model']}): test Macro-F1 {row['test_macro_f1']:.4f}, balanced accuracy {row['test_balanced_accuracy']:.4f}, log-loss {row['test_log_loss']:.4f}. Top dropout SHAP feature: {row['top_dropout_shap_feature']}. Top permutation feature: {row['top_permutation_feature']}."
        )
    stage_c = pd.concat(all_shap_dropout, ignore_index=True)
    stage_c_top = stage_c[stage_c["experiment"] == "Stage C tuned"].head(10)["feature"].tolist()
    full_top = stage_c[stage_c["experiment"] == "Full feature tuned"].head(10)["feature"].tolist()
    abl_top = stage_c[stage_c["experiment"] == "Academic + demographic-financial tuned"].head(10)["feature"].tolist()
    common_key = sorted(set(stage_c_top) & set(full_top) & set(abl_top))
    report_lines.extend(
        [
            "",
            "## Policy Interpretation",
            "",
            "The main paper interpretation should emphasize variables that appear repeatedly across the Stage C, full-feature, and academic + demographic/financial models. Repeated features are more reliable than one-model-only signals. The strongest common dropout-risk structure is expected around semester performance, tuition/financial vulnerability, and student profile variables.",
            "",
            f"Common top-10 dropout SHAP features across Stage C, full-feature, and academic + demographic/financial models: {', '.join(common_key) if common_key else 'none in the strict top-10 intersection'}.",
            "",
            "Use the dropout-class SHAP plots when discussing dropout risk specifically. Use the global SHAP plots when discussing overall multiclass model behavior. Use permutation importance as a robustness check because it measures held-out performance degradation when each raw feature is disrupted.",
        ]
    )
    (REPORT_DIR / "feature_importance_and_shap_policy_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    status = {
        "experiments": len(experiments_df),
        "permutation_repeats": 10,
        "top_n_for_paper": 10,
        "dropout_class": "Dropout",
    }
    (OUT_DIR / "stage7_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
