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

from models.feature_importance_analysis import (
    PREP_DIR,
    OUT_DIR,
    build_preprocessor,
    encoded_to_raw_feature,
    model_label_context,
    predict_proba_aligned,
    selected_experiments,
)

warnings.filterwarnings("ignore")


SHAP2_DIR = OUT_DIR / "SHAP2"
CSV_DIR = SHAP2_DIR / "csv"
REPORT_DIR = SHAP2_DIR / "reports"
PLOT_COLOR = "#59A14F"
TOP_N = 10


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


def get_shap_values(model, x: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x)
    if isinstance(values, list):
        return np.stack(values, axis=2)
    values = np.asarray(values)
    if values.ndim == 2:
        return values[:, :, None]
    return values


def aggregate_encoded_to_raw(
    shap_values: np.ndarray,
    encoded_features: list[str],
    raw_features: list[str],
) -> tuple[np.ndarray, list[str]]:
    groups = []
    for feature in raw_features:
        cols = [
            idx
            for idx, encoded in enumerate(encoded_features)
            if encoded == feature or encoded.startswith(f"{feature}_")
        ]
        if cols:
            groups.append((feature, cols))

    aggregated = np.zeros((shap_values.shape[0], len(groups), shap_values.shape[2]), dtype=float)
    for out_idx, (_, cols) in enumerate(groups):
        aggregated[:, out_idx, :] = shap_values[:, cols, :].sum(axis=1)
    return aggregated, [name for name, _ in groups]


def raw_feature_value_matrix(raw_df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    values = []
    for feature in feature_names:
        series = raw_df[feature]
        if pd.api.types.is_numeric_dtype(series):
            values.append(pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy(dtype=float))
        else:
            codes = pd.Categorical(series.astype("string").fillna("__MISSING__")).codes.astype(float)
            values.append(codes)
    return np.column_stack(values)


def shap_bar(values: np.ndarray, features: list[str], out_path: Path, title: str) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": features, "mean_abs_shap": np.mean(np.abs(values), axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    top = importance.head(TOP_N).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=PLOT_COLOR)
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return importance


def shap_beeswarm(values: np.ndarray, feature_values: np.ndarray, features: list[str], out_path: Path, title: str) -> None:
    importance_order = np.argsort(np.mean(np.abs(values), axis=0))[::-1][:TOP_N]
    explanation = shap.Explanation(
        values=values[:, importance_order],
        data=feature_values[:, importance_order],
        feature_names=[features[i] for i in importance_order],
    )
    plt.figure(figsize=(9.5, 6.6))
    shap.plots.beeswarm(explanation, max_display=TOP_N, show=False, color=plt.get_cmap("viridis"))
    ax = plt.gca()
    ax.set_title(title, fontsize=13, pad=12)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def shap_waterfall(
    values: np.ndarray,
    base_value: float,
    feature_values: np.ndarray,
    features: list[str],
    sample_idx: int,
    out_path: Path,
    title: str,
) -> None:
    importance_order = np.argsort(np.abs(values[sample_idx]))[::-1][:TOP_N]
    explanation = shap.Explanation(
        values=values[sample_idx, importance_order],
        base_values=base_value,
        data=feature_values[sample_idx, importance_order],
        feature_names=[features[i] for i in importance_order],
    )
    plt.figure(figsize=(9.5, 6.8))
    shap.plots.waterfall(explanation, max_display=TOP_N, show=False)
    ax = plt.gca()
    ax.set_title(title, fontsize=13, pad=12)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    mkdirs(SHAP2_DIR, CSV_DIR, REPORT_DIR)

    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")
    target = metadata["targets"]["multiclass"]
    class_names = sorted(df[target].unique())
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    summary_rows = []
    all_global = []
    all_dropout = []

    for exp in selected_experiments(metadata):
        exp_slug = slug(exp["experiment_key"])
        exp_dir = SHAP2_DIR / exp_slug
        mkdirs(exp_dir)

        with open(exp["model_path"], "rb") as f:
            model = pickle.load(f)

        y_by_split, labels, dropout_idx = model_label_context(model, df, target, class_to_id)
        preprocessor, split_frames, x_arrays = build_preprocessor(
            df, exp["features"], metadata, exp["feature_order"]
        )
        x_test_df = pd.DataFrame(x_arrays["test"], columns=preprocessor.output_columns_)
        shap_values_encoded = get_shap_values(model, x_test_df)
        shap_values_raw, raw_feature_names = aggregate_encoded_to_raw(
            shap_values_encoded, preprocessor.output_columns_, exp["features"]
        )
        feature_values = raw_feature_value_matrix(split_frames["test"], raw_feature_names)

        dropout_values = shap_values_raw[:, :, dropout_idx]
        global_values = np.mean(np.abs(shap_values_raw), axis=2)

        global_importance = shap_bar(
            global_values,
            raw_feature_names,
            exp_dir / "global_shap_bar_top10.png",
            f"{exp['display_name']}: Global SHAP Importance",
        )
        dropout_importance = shap_bar(
            dropout_values,
            raw_feature_names,
            exp_dir / "dropout_class_shap_bar_top10.png",
            f"{exp['display_name']}: Dropout-Class SHAP Importance",
        )
        shap_beeswarm(
            dropout_values,
            feature_values,
            raw_feature_names,
            exp_dir / "dropout_class_shap_beeswarm_top10.png",
            f"{exp['display_name']}: Dropout-Class SHAP Summary",
        )
        shap_beeswarm(
            global_values,
            feature_values,
            raw_feature_names,
            exp_dir / "global_shap_beeswarm_top10.png",
            f"{exp['display_name']}: Global SHAP Summary",
        )

        proba = predict_proba_aligned(model, x_arrays["test"], labels)
        high_risk_idx = int(np.argmax(proba[:, dropout_idx])) if proba is not None else 0
        base_values = shap.TreeExplainer(model).expected_value
        if isinstance(base_values, list):
            base_value = float(base_values[dropout_idx])
        else:
            base_arr = np.asarray(base_values)
            base_value = float(base_arr[dropout_idx]) if base_arr.ndim else float(base_arr)
        shap_waterfall(
            dropout_values,
            base_value,
            feature_values,
            raw_feature_names,
            high_risk_idx,
            exp_dir / "dropout_high_risk_waterfall_top10.png",
            f"{exp['display_name']}: High-Risk Dropout Explanation",
        )

        global_importance.insert(0, "experiment", exp["display_name"])
        global_importance.insert(1, "experiment_key", exp["experiment_key"])
        dropout_importance.insert(0, "experiment", exp["display_name"])
        dropout_importance.insert(1, "experiment_key", exp["experiment_key"])
        global_importance.to_csv(exp_dir / "global_shap_bar_top10_source.csv", index=False)
        dropout_importance.to_csv(exp_dir / "dropout_class_shap_bar_top10_source.csv", index=False)
        all_global.append(global_importance)
        all_dropout.append(dropout_importance)

        summary_rows.append(
            {
                "experiment": exp["display_name"],
                "experiment_key": exp["experiment_key"],
                "model": exp["model"],
                "top_global_shap_feature": global_importance.iloc[0]["feature"],
                "top_dropout_shap_feature": dropout_importance.iloc[0]["feature"],
                "high_risk_test_row_position": high_risk_idx,
                "high_risk_dropout_probability": float(proba[high_risk_idx, dropout_idx]) if proba is not None else np.nan,
                "dropout_beeswarm_plot": str(exp_dir / "dropout_class_shap_beeswarm_top10.png"),
                "dropout_waterfall_plot": str(exp_dir / "dropout_high_risk_waterfall_top10.png"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(CSV_DIR / "shap2_academic_plot_summary.csv", index=False)
    pd.concat(all_global, ignore_index=True).to_csv(CSV_DIR / "shap2_global_bar_source_all_models.csv", index=False)
    pd.concat(all_dropout, ignore_index=True).to_csv(CSV_DIR / "shap2_dropout_bar_source_all_models.csv", index=False)

    report = [
        "# SHAP2 Academic Plot Report",
        "",
        "This folder contains publication-oriented SHAP graphics for the five selected tuned models. Each model has four main plots: global SHAP bar, global SHAP beeswarm, Dropout-class SHAP bar, Dropout-class SHAP beeswarm, plus a high-risk student waterfall explanation for the Dropout class.",
        "",
        "Recommended paper figures: use the Dropout-class SHAP beeswarm for Stage C or the full-feature model, and use the Dropout-class SHAP bar for the academic + demographic/financial model because it communicates the policy story more directly.",
        "",
        "Top SHAP2 dropout features by model:",
    ]
    for _, row in summary.iterrows():
        report.append(f"- {row['experiment']}: {row['top_dropout_shap_feature']}")
    (REPORT_DIR / "shap2_academic_plot_report.md").write_text("\n".join(report), encoding="utf-8")

    status = {
        "experiments": int(len(summary)),
        "plots_per_experiment": 5,
        "top_n": TOP_N,
    }
    (SHAP2_DIR / "shap2_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
