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


PREP_DIR = Path("results") / "2.Model Processing"
OUT_DIR = Path("results") / "6.Ablation Study"

ABLATION_BLOCKS = {
    "academic_only": "1.Academic only",
    "academic_plus_financial_demographic": "2.Academic + demographic-financial",
    "academic_plus_parental": "3.Academic + parental education",
    "academic_plus_macro": "4.Academic + macroeconomic",
    "academic_plus_course_application": "5.Academic + course-application",
}

FOCUSED_GRIDS = {
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


def parameter_grid(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("/", "_")
        .replace("-", "_")
    )


def build_experiment_model(spec, key: str, params: dict | None = None, block_key: str = ""):
    params = dict(params or {})
    if key == "catboost":
        train_dir = Path("models") / "catboost_info" / "ablation" / block_key
        train_dir.mkdir(parents=True, exist_ok=True)
        params.setdefault("train_dir", str(train_dir))
    return spec.build(**params)


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
        rows.append({"split": split, **classification_metrics(y_true, y_pred, y_proba)})
    return rows


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


def save_split_tables(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "log_loss"]
    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split].copy()
        split_df.to_csv(out_dir / f"{prefix}_{split}_metrics_by_model.csv", index=False)
        pivot = split_df.pivot_table(index="model", values=metrics, aggfunc="first")
        pivot.sort_values("macro_f1", ascending=False).to_csv(out_dir / f"{prefix}_{split}_metric_pivot.csv")


def top_importance(model, feature_names: list[str]) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importances_"):
        return None
    values = np.asarray(model.feature_importances_, dtype=float)
    return pd.DataFrame({"feature": feature_names, "importance": values}).sort_values("importance", ascending=False)


def plot_importance(importance: pd.DataFrame, out_path: Path, title: str) -> None:
    top = importance.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top["feature"], top["importance"], color="#59A14F")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_block(block_key: str, folder_name: str, df: pd.DataFrame, metadata: dict, labels: np.ndarray, y_by_split: dict) -> dict:
    block_dir = OUT_DIR / folder_name
    csv_dir = block_dir / "csv"
    plot_dir = block_dir / "plots"
    report_dir = block_dir / "reports"
    model_dir = block_dir / "models"
    tuned_dir = block_dir / "Hyp Tuned"
    tuned_csv_dir = tuned_dir / "csv"
    tuned_plot_dir = tuned_dir / "plots"
    tuned_report_dir = tuned_dir / "reports"
    tuned_model_dir = tuned_dir / "models"
    for directory in [csv_dir, plot_dir, report_dir, model_dir, tuned_csv_dir, tuned_plot_dir, tuned_report_dir, tuned_model_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    target = metadata["targets"]["multiclass"]
    features = metadata["ablation_feature_sets"][block_key]
    numeric_features = [f for f in metadata["feature_types"]["numeric_features"] if f in features]
    categorical_features = [f for f in metadata["feature_types"]["categorical_features"] if f in features]
    splits = {split: df[df["Split"] == split].copy() for split in ["train", "validation", "test"]}

    preprocessor = TabularOneHotScaler(numeric_features=numeric_features, categorical_features=categorical_features)
    x_train_df = preprocessor.fit_transform(splits["train"][features])
    x_by_split_df = {"train": x_train_df}
    for split in ["validation", "test"]:
        x_by_split_df[split] = preprocessor.transform(splits[split][features])
    x_by_split = {split: matrix.to_numpy(dtype=np.float32) for split, matrix in x_by_split_df.items()}

    with open(model_dir / f"{block_key}__preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    shape_row = {
        "ablation_block": block_key,
        "folder": folder_name,
        "raw_features": len(features),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "encoded_features": x_train_df.shape[1],
        "train_rows": len(splits["train"]),
        "validation_rows": len(splits["validation"]),
        "test_rows": len(splits["test"]),
    }
    pd.DataFrame([shape_row]).to_csv(csv_dir / "feature_shape.csv", index=False)

    baseline_rows, baseline_failures = [], []
    for key, spec in MODEL_REGISTRY.items():
        if not spec.is_available():
            baseline_failures.append({"model_key": key, "model": spec.display_name, "error": "model package unavailable"})
            continue
        try:
            start = time.time()
            model = build_experiment_model(spec, key, block_key=block_key)
            model.fit(x_by_split["train"], y_by_split["train"])
            fit_seconds = time.time() - start
            model_path = model_dir / f"{block_key}__{key}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            for row in evaluate_model(model, x_by_split, y_by_split, labels):
                baseline_rows.append(
                    {
                        "ablation_block": block_key,
                        "block_folder": folder_name,
                        "model_key": key,
                        "model": spec.display_name,
                        "raw_features": len(features),
                        "encoded_features": x_train_df.shape[1],
                        "fit_seconds": fit_seconds,
                        "model_path": str(model_path),
                        **row,
                    }
                )
        except Exception as exc:
            baseline_failures.append({"model_key": key, "model": spec.display_name, "error": repr(exc)})

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(csv_dir / "ablation_model_comparison.csv", index=False)
    pd.DataFrame(baseline_failures).to_csv(csv_dir / "ablation_model_failures.csv", index=False)
    save_split_tables(baseline_df, csv_dir, "ablation")
    for metric in ["macro_f1", "balanced_accuracy", "log_loss"]:
        plot_metric_bars(baseline_df, plot_dir, "ablation", metric, f"{folder_name} Baseline")

    tuned_rows, candidate_rows, tuned_failures, best_models = [], [], [], {}
    for key, spec in MODEL_REGISTRY.items():
        if not spec.is_available():
            tuned_failures.append({"model_key": key, "model": spec.display_name, "params": "", "error": "model package unavailable"})
            continue
        best = None
        for candidate_id, params in enumerate(parameter_grid(FOCUSED_GRIDS[key]), start=1):
            try:
                start = time.time()
                model = build_experiment_model(spec, key, params, block_key=block_key)
                model.fit(x_by_split["train"], y_by_split["train"])
                fit_seconds = time.time() - start
                val_metrics = evaluate_model(
                    model,
                    {"validation": x_by_split["validation"]},
                    {"validation": y_by_split["validation"]},
                    labels,
                )[0]
                candidate_rows.append(
                    {
                        "ablation_block": block_key,
                        "block_folder": folder_name,
                        "model_key": key,
                        "model": spec.display_name,
                        "candidate_id": candidate_id,
                        "params": json.dumps(params),
                        "fit_seconds": fit_seconds,
                        **val_metrics,
                    }
                )
                score = (val_metrics["macro_f1"], val_metrics["balanced_accuracy"], -val_metrics.get("log_loss", np.inf))
                if best is None or score > best["score"]:
                    best = {"score": score, "params": params, "model": model, "fit_seconds": fit_seconds}
            except Exception as exc:
                tuned_failures.append({"model_key": key, "model": spec.display_name, "params": json.dumps(params), "error": repr(exc)})
        if best is None:
            continue
        best_model = best["model"]
        best_models[key] = best_model
        model_path = tuned_model_dir / f"{block_key}__{key}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        for row in evaluate_model(best_model, x_by_split, y_by_split, labels):
            tuned_rows.append(
                {
                    "ablation_block": block_key,
                    "block_folder": folder_name,
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
    candidate_df.to_csv(tuned_csv_dir / "ablation_tuning_candidate_validation_results.csv", index=False)
    tuned_df.to_csv(tuned_csv_dir / "ablation_tuned_model_comparison.csv", index=False)
    pd.DataFrame(tuned_failures).to_csv(tuned_csv_dir / "ablation_tuning_failures.csv", index=False)
    save_split_tables(tuned_df, tuned_csv_dir, "ablation_tuned")
    for metric in ["macro_f1", "balanced_accuracy", "log_loss"]:
        plot_metric_bars(tuned_df, tuned_plot_dir, "ablation_tuned", metric, f"{folder_name} Tuned")

    validation_ranking = baseline_df[baseline_df["split"] == "validation"].sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True]
    )
    tuned_validation_ranking = tuned_df[tuned_df["split"] == "validation"].sort_values(
        ["macro_f1", "balanced_accuracy", "log_loss"], ascending=[False, False, True]
    )
    validation_ranking.to_csv(csv_dir / "ablation_validation_model_ranking.csv", index=False)
    tuned_validation_ranking.to_csv(tuned_csv_dir / "ablation_tuned_validation_model_ranking.csv", index=False)

    comparison = baseline_df.merge(
        tuned_df,
        on=["ablation_block", "block_folder", "model_key", "model", "split"],
        how="outer",
        suffixes=("_before_tuning", "_after_tuning"),
    )
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "macro_precision", "macro_recall", "log_loss"]:
        comparison[f"{metric}_change_after_minus_before"] = comparison[f"{metric}_after_tuning"] - comparison[f"{metric}_before_tuning"]
    comparison.to_csv(tuned_csv_dir / "ablation_before_after_tuning_comparison.csv", index=False)

    if len(tuned_validation_ranking):
        best = tuned_validation_ranking.iloc[0]
        best_model = best_models.get(best["model_key"])
        if best_model is not None:
            importance = top_importance(best_model, preprocessor.output_columns_)
            if importance is not None:
                importance.to_csv(tuned_csv_dir / "best_tuned_ablation_feature_importance.csv", index=False)
                plot_importance(importance, tuned_plot_dir / "best_tuned_ablation_feature_importance_top20.png", f"{folder_name}: Top Features")

    report_lines = [
        f"# Ablation Block Report: {folder_name}",
        "",
        f"This block uses {len(features)} raw predictors transformed into {x_train_df.shape[1]} encoded/scaled features.",
        "",
    ]
    if len(validation_ranking):
        row = validation_ranking.iloc[0]
        report_lines.append(
            f"Best baseline validation model: **{row['model']}** with Macro-F1 = {row['macro_f1']:.4f}, balanced accuracy = {row['balanced_accuracy']:.4f}, and log-loss = {row['log_loss']:.4f}."
        )
    if len(tuned_validation_ranking):
        row = tuned_validation_ranking.iloc[0]
        test_row = tuned_df[(tuned_df["model_key"] == row["model_key"]) & (tuned_df["split"] == "test")].iloc[0]
        report_lines.extend(
            [
                "",
                f"Best tuned validation model: **{row['model']}** with Macro-F1 = {row['macro_f1']:.4f}, balanced accuracy = {row['balanced_accuracy']:.4f}, and log-loss = {row['log_loss']:.4f}.",
                f"On test, the validation-selected tuned model reached Macro-F1 = {test_row['macro_f1']:.4f}, balanced accuracy = {test_row['balanced_accuracy']:.4f}, and log-loss = {test_row['log_loss']:.4f}.",
            ]
        )
    (report_dir / "ablation_block_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        **shape_row,
        "baseline_rows": len(baseline_df),
        "tuning_candidates": len(candidate_df),
        "tuned_rows": len(tuned_df),
        "baseline_failures": len(baseline_failures),
        "tuning_failures": len(tuned_failures),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")
    target = metadata["targets"]["multiclass"]

    class_names = sorted(df[target].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    labels = np.array(list(range(len(class_names))))
    y_by_split = {
        split: df[df["Split"] == split][target].map(class_to_id).to_numpy(dtype=int)
        for split in ["train", "validation", "test"]
    }
    pd.DataFrame(
        [{"target_class": name, "encoded_label": idx} for name, idx in class_to_id.items()]
    ).to_csv(OUT_DIR / "target_label_mapping.csv", index=False)

    status_rows = []
    baseline_all, tuned_all, comparison_all = [], [], []
    for block_key, folder_name in ABLATION_BLOCKS.items():
        status_rows.append(run_block(block_key, folder_name, df, metadata, labels, y_by_split))
        block_dir = OUT_DIR / folder_name
        baseline_all.append(pd.read_csv(block_dir / "csv" / "ablation_model_comparison.csv"))
        tuned_all.append(pd.read_csv(block_dir / "Hyp Tuned" / "csv" / "ablation_tuned_model_comparison.csv"))
        comparison_all.append(pd.read_csv(block_dir / "Hyp Tuned" / "csv" / "ablation_before_after_tuning_comparison.csv"))

    aggregate_csv_dir = OUT_DIR / "csv"
    aggregate_plot_dir = OUT_DIR / "plots"
    aggregate_report_dir = OUT_DIR / "reports"
    for directory in [aggregate_csv_dir, aggregate_plot_dir, aggregate_report_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    status_df = pd.DataFrame(status_rows)
    baseline_df = pd.concat(baseline_all, ignore_index=True)
    tuned_df = pd.concat(tuned_all, ignore_index=True)
    comparison_df = pd.concat(comparison_all, ignore_index=True)
    status_df.to_csv(aggregate_csv_dir / "ablation_run_status.csv", index=False)
    baseline_df.to_csv(aggregate_csv_dir / "all_ablation_baseline_model_comparison.csv", index=False)
    tuned_df.to_csv(aggregate_csv_dir / "all_ablation_tuned_model_comparison.csv", index=False)
    comparison_df.to_csv(aggregate_csv_dir / "all_ablation_before_after_tuning_comparison.csv", index=False)

    baseline_best = baseline_df[baseline_df["split"] == "validation"].sort_values(
        ["ablation_block", "macro_f1", "balanced_accuracy", "log_loss"],
        ascending=[True, False, False, True],
    ).groupby("ablation_block", as_index=False).first()
    tuned_best = tuned_df[tuned_df["split"] == "validation"].sort_values(
        ["ablation_block", "macro_f1", "balanced_accuracy", "log_loss"],
        ascending=[True, False, False, True],
    ).groupby("ablation_block", as_index=False).first()
    baseline_best.to_csv(aggregate_csv_dir / "best_baseline_validation_model_by_ablation_block.csv", index=False)
    tuned_best.to_csv(aggregate_csv_dir / "best_tuned_validation_model_by_ablation_block.csv", index=False)

    paper_table = tuned_best.merge(
        tuned_df[tuned_df["split"] == "test"],
        on=["ablation_block", "model_key", "model"],
        suffixes=("_validation", "_test"),
    )
    paper_table = paper_table[
        [
            "ablation_block",
            "block_folder_validation",
            "model",
            "raw_features_validation",
            "encoded_features_validation",
            "macro_f1_validation",
            "balanced_accuracy_validation",
            "log_loss_validation",
            "macro_f1_test",
            "balanced_accuracy_test",
            "log_loss_test",
        ]
    ].rename(columns={"block_folder_validation": "ablation_folder"})
    paper_table.to_csv(aggregate_csv_dir / "paper_ready_best_tuned_ablation_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    plot_df = paper_table.sort_values("macro_f1_test", ascending=True)
    ax.barh(plot_df["ablation_folder"], plot_df["macro_f1_test"], color="#4C78A8")
    ax.set_xlabel("Test Macro-F1")
    ax.set_title("Best Tuned Model by Ablation Block")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(aggregate_plot_dir / "best_tuned_ablation_blocks_test_macro_f1.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# Ablation Study Summary Report",
        "",
        "This ablation study compares whether additional variable blocks improve dropout prediction beyond academic performance. Each block was trained with the same train/validation/test split and the same encoded/scaled input protocol.",
        "",
        "## Best Tuned Validation Models",
        "",
    ]
    for _, row in paper_table.sort_values("macro_f1_validation", ascending=False).iterrows():
        report_lines.append(
            f"- {row['ablation_folder']}: {row['model']} | validation Macro-F1 {row['macro_f1_validation']:.4f}; test Macro-F1 {row['macro_f1_test']:.4f}; test log-loss {row['log_loss_test']:.4f}."
        )
    best_test = paper_table.sort_values(["macro_f1_test", "balanced_accuracy_test", "log_loss_test"], ascending=[False, False, True]).iloc[0]
    report_lines.extend(
        [
            "",
            "## Paper Interpretation",
            "",
            f"The strongest ablation block on the held-out test set was **{best_test['ablation_folder']}**, selected by validation performance and tested independently. This tells us which information block adds the most practical value after academic signals.",
            "",
            "For the paper, use the `paper_ready_best_tuned_ablation_table.csv` table and the `best_tuned_ablation_blocks_test_macro_f1.png` plot. These directly answer whether demographic/financial, parental, macroeconomic, or course/application information improves prediction after academic performance.",
        ]
    )
    (aggregate_report_dir / "ablation_study_summary_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    status = {
        "ablation_blocks": len(ABLATION_BLOCKS),
        "baseline_rows": int(len(baseline_df)),
        "tuned_rows": int(len(tuned_df)),
        "tuning_candidates": int(sum(status_df["tuning_candidates"])),
        "baseline_failures": int(sum(status_df["baseline_failures"])),
        "tuning_failures": int(sum(status_df["tuning_failures"])),
    }
    (OUT_DIR / "stage6_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
