from __future__ import annotations

import itertools
import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.evaluation import classification_metrics
from models.model_registry import MODEL_REGISTRY
from models.preprocessing import TabularOneHotScaler

warnings.filterwarnings("ignore")


RANDOM_STATE = 123
PREP_DIR = Path("results") / "2.Model Processing"
STAGED_DIR = Path("results") / "3.Staged Early Warning Modeling"
TUNING_DIR = Path("results") / "4.Hyperparameter Tuning"
OUT_DIR = Path("results") / "5.Full Feature Final Modeling"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
TUNED_DIR = OUT_DIR / "Hyp Tuned"
TUNED_CSV_DIR = TUNED_DIR / "csv"
TUNED_PLOT_DIR = TUNED_DIR / "plots"
TUNED_REPORT_DIR = TUNED_DIR / "reports"
TUNED_MODEL_DIR = TUNED_DIR / "models"

for directory in [
    CSV_DIR,
    PLOT_DIR,
    REPORT_DIR,
    MODEL_DIR,
    TUNED_CSV_DIR,
    TUNED_PLOT_DIR,
    TUNED_REPORT_DIR,
    TUNED_MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def parameter_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


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


def evaluate_model(model, x_by_split, y_by_split, labels) -> list[dict]:
    rows = []
    for split, x in x_by_split.items():
        y_true = y_by_split[split]
        y_pred = np.asarray(model.predict(x)).ravel()
        y_proba = predict_proba_aligned(model, x, labels)
        metrics = classification_metrics(y_true, y_pred, y_proba)
        rows.append({"split": split, **metrics})
    return rows


def build_experiment_model(spec, key: str, params: dict | None = None):
    params = dict(params or {})
    if key == "catboost":
        params.setdefault("train_dir", str(Path("models") / "catboost_info"))
    return spec.build(**params)


def save_pivot_tables(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "log_loss"]
    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split].copy()
        split_df.to_csv(out_dir / f"{prefix}_{split}_metrics_by_model.csv", index=False)
        pivot = split_df.pivot_table(index="model", values=metrics, aggfunc="first")
        pivot = pivot.sort_values("macro_f1", ascending=False)
        pivot.to_csv(out_dir / f"{prefix}_{split}_metric_pivot.csv")


def plot_metric_bars(df: pd.DataFrame, out_dir: Path, prefix: str, metric: str, title: str) -> None:
    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split].sort_values(metric, ascending=False)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh(split_df["model"], split_df[metric], color="#4C78A8")
        ax.invert_yaxis()
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(f"{title}: {split.title()}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{split}_{metric}.png", dpi=180)
        plt.close(fig)


def model_feature_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.mean(np.abs(np.asarray(model.coef_, dtype=float)), axis=0)
    else:
        return None
    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(importance: pd.DataFrame, out_path: Path, title: str, top_n: int = 25) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance"], color="#59A14F")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")

    target = metadata["targets"]["multiclass"]
    features = metadata["ablation_feature_sets"]["full"]
    numeric_features = [f for f in metadata["feature_types"]["numeric_features"] if f in features]
    categorical_features = [f for f in metadata["feature_types"]["categorical_features"] if f in features]

    splits = {
        split: df[df["Split"] == split].copy()
        for split in ["train", "validation", "test"]
    }
    class_names = sorted(df[target].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    target_mapping = pd.DataFrame(
        [{"target_class": name, "encoded_label": idx} for name, idx in class_to_id.items()]
    )
    target_mapping.to_csv(CSV_DIR / "target_label_mapping.csv", index=False)
    target_mapping.to_csv(TUNED_CSV_DIR / "target_label_mapping.csv", index=False)
    y_by_split = {split: frame[target].map(class_to_id).to_numpy(dtype=int) for split, frame in splits.items()}
    labels = np.array(list(range(len(class_names))))

    preprocessor = TabularOneHotScaler(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    x_train_df = preprocessor.fit_transform(splits["train"][features])
    x_by_split_df = {"train": x_train_df}
    for split in ["validation", "test"]:
        x_by_split_df[split] = preprocessor.transform(splits[split][features])
    x_by_split = {split: matrix.to_numpy(dtype=np.float32) for split, matrix in x_by_split_df.items()}

    with open(MODEL_DIR / "full_feature_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    feature_shape = pd.DataFrame(
        [
            {
                "feature_set": "full",
                "raw_features": len(features),
                "numeric_features": len(numeric_features),
                "categorical_features": len(categorical_features),
                "encoded_features": x_train_df.shape[1],
                "train_rows": len(splits["train"]),
                "validation_rows": len(splits["validation"]),
                "test_rows": len(splits["test"]),
            }
        ]
    )
    feature_shape.to_csv(CSV_DIR / "full_feature_encoded_shape.csv", index=False)

    baseline_rows = []
    fitted_models = {}
    failures = []
    for key, spec in MODEL_REGISTRY.items():
        if not spec.is_available():
            failures.append({"model_key": key, "model": spec.display_name, "error": "model package unavailable"})
            continue
        try:
            start = time.time()
            model = build_experiment_model(spec, key)
            model.fit(x_by_split["train"], y_by_split["train"])
            fit_seconds = time.time() - start
            fitted_models[key] = model
            with open(MODEL_DIR / f"full__{key}.pkl", "wb") as f:
                pickle.dump(model, f)
            for row in evaluate_model(model, x_by_split, y_by_split, labels):
                baseline_rows.append(
                    {
                        "feature_set": "full",
                        "model_key": key,
                        "model": spec.display_name,
                        "raw_features": len(features),
                        "encoded_features": x_train_df.shape[1],
                        "fit_seconds": fit_seconds,
                        "model_path": str(MODEL_DIR / f"full__{key}.pkl"),
                        **row,
                    }
                )
        except Exception as exc:
            failures.append({"model_key": key, "model": spec.display_name, "error": repr(exc)})

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(CSV_DIR / "full_feature_model_comparison.csv", index=False)
    pd.DataFrame(failures).to_csv(CSV_DIR / "full_feature_model_failures.csv", index=False)
    save_pivot_tables(baseline_df, CSV_DIR, "full_feature")
    for metric in ["macro_f1", "balanced_accuracy", "log_loss"]:
        plot_metric_bars(baseline_df, PLOT_DIR, "full_feature", metric, "Full-Feature Baseline Models")

    val_baseline = baseline_df[baseline_df["split"] == "validation"].sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True]
    )
    val_baseline.to_csv(CSV_DIR / "full_feature_validation_model_ranking.csv", index=False)

    focused_grids = {
        "decision_tree": {
            "max_depth": [5, 8, 12, None],
            "min_samples_leaf": [5, 10, 20],
            "criterion": ["gini", "entropy"],
        },
        "random_forest": {
            "n_estimators": [300, 500],
            "max_depth": [None, 12, 18],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt"],
        },
        "xgboost": {
            "n_estimators": [300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        },
        "lightgbm": {
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31],
            "max_depth": [-1, 8],
        },
        "catboost": {
            "iterations": [300, 500],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6],
            "l2_leaf_reg": [1, 3],
        },
        "rbf_svm": {
            "C": [0.5, 1.0, 3.0, 10.0],
            "gamma": ["scale", 0.01, 0.03],
        },
        "mlp": {
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001],
            "learning_rate_init": [0.0005, 0.001],
        },
    }

    candidate_rows = []
    tuned_rows = []
    tuned_failures = []
    best_models = {}

    for key, spec in MODEL_REGISTRY.items():
        if not spec.is_available():
            tuned_failures.append({"model_key": key, "model": spec.display_name, "params": "", "error": "model package unavailable"})
            continue
        candidates = parameter_grid(focused_grids[key])
        best = None
        for candidate_id, params in enumerate(candidates, start=1):
            try:
                start = time.time()
                model = build_experiment_model(spec, key, params)
                model.fit(x_by_split["train"], y_by_split["train"])
                fit_seconds = time.time() - start
                validation_metrics = evaluate_model(
                    model,
                    {"validation": x_by_split["validation"]},
                    {"validation": y_by_split["validation"]},
                    labels,
                )[0]
                row = {
                    "feature_set": "full",
                    "model_key": key,
                    "model": spec.display_name,
                    "candidate_id": candidate_id,
                    "params": json.dumps(params),
                    "fit_seconds": fit_seconds,
                    **validation_metrics,
                }
                candidate_rows.append(row)
                score = (
                    validation_metrics["macro_f1"],
                    validation_metrics["balanced_accuracy"],
                    -validation_metrics.get("log_loss", np.inf),
                )
                if best is None or score > best["score"]:
                    best = {"score": score, "params": params, "model": model, "fit_seconds": fit_seconds}
            except Exception as exc:
                tuned_failures.append(
                    {
                        "model_key": key,
                        "model": spec.display_name,
                        "params": json.dumps(params),
                        "error": repr(exc),
                    }
                )
        if best is None:
            continue

        best_model = best["model"]
        best_models[key] = best_model
        model_path = TUNED_MODEL_DIR / f"full__{key}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        for row in evaluate_model(best_model, x_by_split, y_by_split, labels):
            tuned_rows.append(
                {
                    "feature_set": "full",
                    "model_key": key,
                    "model": spec.display_name,
                    "raw_features": len(features),
                    "encoded_features": x_train_df.shape[1],
                    "best_params": json.dumps(best["params"]),
                    "fit_seconds": best["fit_seconds"],
                    "model_path": str(model_path),
                    **row,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    tuned_df = pd.DataFrame(tuned_rows)
    candidate_df.to_csv(TUNED_CSV_DIR / "full_feature_tuning_candidate_validation_results.csv", index=False)
    tuned_df.to_csv(TUNED_CSV_DIR / "full_feature_tuned_model_comparison.csv", index=False)
    pd.DataFrame(tuned_failures).to_csv(TUNED_CSV_DIR / "full_feature_tuning_failures.csv", index=False)
    save_pivot_tables(tuned_df, TUNED_CSV_DIR, "full_feature_tuned")
    for metric in ["macro_f1", "balanced_accuracy", "log_loss"]:
        plot_metric_bars(tuned_df, TUNED_PLOT_DIR, "full_feature_tuned", metric, "Full-Feature Tuned Models")

    val_tuned = tuned_df[tuned_df["split"] == "validation"].sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True]
    )
    val_tuned.to_csv(TUNED_CSV_DIR / "full_feature_tuned_validation_model_ranking.csv", index=False)

    comparison = baseline_df.merge(
        tuned_df,
        on=["feature_set", "model_key", "model", "split"],
        how="outer",
        suffixes=("_before_tuning", "_after_tuning"),
    )
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "log_loss"]:
        comparison[f"{metric}_change_after_minus_before"] = (
            comparison[f"{metric}_after_tuning"] - comparison[f"{metric}_before_tuning"]
        )
    comparison.to_csv(TUNED_CSV_DIR / "full_feature_before_after_tuning_comparison.csv", index=False)

    before_best = val_baseline.iloc[0] if len(val_baseline) else None
    after_best = val_tuned.iloc[0] if len(val_tuned) else None

    if after_best is not None:
        best_key = after_best["model_key"]
        best_model = best_models[best_key]
        importance = model_feature_importance(best_model, preprocessor.output_columns_)
        if importance is not None:
            importance.to_csv(TUNED_CSV_DIR / "best_tuned_full_feature_importance.csv", index=False)
            plot_importance(
                importance,
                TUNED_PLOT_DIR / "best_tuned_full_feature_importance_top25.png",
                f"Top 25 Features: {after_best['model']}",
            )

    stage_c_before = pd.read_csv(STAGED_DIR / "csv" / "staged_model_comparison.csv")
    stage_c_tuned = pd.read_csv(TUNING_DIR / "csv" / "tuned_model_comparison.csv")
    stage_c_before = stage_c_before[(stage_c_before["stage"] == "C_second_semester") & (stage_c_before["split"] == "test")]
    stage_c_tuned = stage_c_tuned[(stage_c_tuned["stage"] == "C_second_semester") & (stage_c_tuned["split"] == "test")]
    full_test_before = baseline_df[baseline_df["split"] == "test"]
    full_test_after = tuned_df[tuned_df["split"] == "test"]
    cross_rows = []
    for label, frame in [
        ("Stage C baseline", stage_c_before),
        ("Stage C tuned", stage_c_tuned),
        ("Full feature baseline", full_test_before),
        ("Full feature tuned", full_test_after),
    ]:
        if len(frame):
            row = frame.sort_values(["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True]).iloc[0]
            cross_rows.append(
                {
                    "comparison_group": label,
                    "best_model": row["model"],
                    "test_macro_f1": row["macro_f1"],
                    "test_balanced_accuracy": row["balanced_accuracy"],
                    "test_log_loss": row["log_loss"],
                }
            )
    pd.DataFrame(cross_rows).to_csv(TUNED_CSV_DIR / "stage_c_vs_full_feature_test_best_models.csv", index=False)

    gain_summary = (
        comparison[comparison["split"].isin(["train", "validation", "test"])]
        .loc[:, ["model", "split", "macro_f1_change_after_minus_before", "balanced_accuracy_change_after_minus_before", "log_loss_change_after_minus_before"]]
        .sort_values(["split", "macro_f1_change_after_minus_before"], ascending=[True, False])
    )
    gain_summary.to_csv(TUNED_CSV_DIR / "full_feature_tuning_gain_summary.csv", index=False)

    report_lines = [
        "# Full-Feature Final Modeling Report",
        "",
        "This analysis trains one final full-information model family using all available predictors from the prepared modeling dataset. It is not a replacement for the staged early-warning analysis. Instead, it gives a maximum-information benchmark that shows how much predictive performance is available when pre-enrollment, first-semester, and second-semester variables are all used together.",
        "",
        f"The full-feature matrix uses {len(features)} raw predictors, transformed into {x_train_df.shape[1]} encoded/scaled columns using training-set preprocessing only.",
        "",
    ]
    if before_best is not None:
        report_lines.extend(
            [
                "## Best Baseline Model",
                "",
                f"Before tuning, the best validation model was **{before_best['model']}** with validation Macro-F1 = {before_best['macro_f1']:.4f}, balanced accuracy = {before_best['balanced_accuracy']:.4f}, and log-loss = {before_best['log_loss']:.4f}.",
                "",
            ]
        )
    if after_best is not None:
        test_after = tuned_df[(tuned_df["model_key"] == after_best["model_key"]) & (tuned_df["split"] == "test")].iloc[0]
        report_lines.extend(
            [
                "## Best Tuned Model",
                "",
                f"After tuning, the best validation model was **{after_best['model']}** with validation Macro-F1 = {after_best['macro_f1']:.4f}, balanced accuracy = {after_best['balanced_accuracy']:.4f}, and log-loss = {after_best['log_loss']:.4f}. On the held-out test set, this same selected model reached Macro-F1 = {test_after['macro_f1']:.4f}, balanced accuracy = {test_after['balanced_accuracy']:.4f}, and log-loss = {test_after['log_loss']:.4f}.",
                "",
            ]
        )
    report_lines.extend(
        [
            "## What To Say In The Paper",
            "",
            "The staged models should be presented as the early-warning contribution, while the full-feature model should be presented as the final maximum-information benchmark. If the full-feature model only slightly improves over Stage C, that supports the argument that second-semester academic and financial signals already capture most dropout risk. If the full-feature model improves meaningfully, then the additional variables should be discussed as useful for end-of-year or final-risk assessment rather than true early warning.",
            "",
            "Recommended tables/figures: full-feature train/validation/test metrics table, before/after tuning comparison table, Stage C vs full-feature best-test comparison, and the top-25 feature-importance plot for the best tuned full-feature model.",
        ]
    )
    (REPORT_DIR / "full_feature_final_modeling_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    tuning_report = [
        "# Full-Feature Hyperparameter Tuning Report",
        "",
        f"Tuning evaluated {len(candidate_df)} candidate model configurations across the seven model families. Selection used validation Macro-F1 first, then validation balanced accuracy, then lower validation log-loss.",
        "",
        "The tuned outputs are saved separately inside `results/5.Full Feature Final Modeling/Hyp Tuned/` to keep baseline and tuned final-model results easy to compare.",
        "",
    ]
    if after_best is not None:
        tuning_report.append(
            f"Best tuned validation model: **{after_best['model']}**; validation Macro-F1 = {after_best['macro_f1']:.4f}, validation balanced accuracy = {after_best['balanced_accuracy']:.4f}, validation log-loss = {after_best['log_loss']:.4f}."
        )
    (TUNED_REPORT_DIR / "full_feature_hyperparameter_tuning_report.md").write_text("\n".join(tuning_report), encoding="utf-8")

    notebook_status = {
        "baseline_rows": int(len(baseline_df)),
        "tuning_candidates": int(len(candidate_df)),
        "tuned_rows": int(len(tuned_df)),
        "baseline_failures": int(len(failures)),
        "tuning_failures": int(len(tuned_failures)),
    }
    (OUT_DIR / "stage5_run_status.json").write_text(json.dumps(notebook_status, indent=2), encoding="utf-8")
    print(json.dumps(notebook_status, indent=2))


if __name__ == "__main__":
    main()
