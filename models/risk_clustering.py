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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from models.preprocessing import TabularOneHotScaler

warnings.filterwarnings("ignore")


PREP_DIR = Path("results") / "2.Model Processing"
FULL_MODEL_DIR = Path("results") / "5.Full Feature Final Modeling" / "Hyp Tuned" / "models"
OUT_DIR = Path("results") / "8.Risk Clustering"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"

RANDOM_STATE = 123


RISK_FEATURES = [
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Age at enrollment",
    "Displaced",
    "Gender",
    "Course",
    "Application mode",
    "Admission grade",
    "Previous qualification (grade)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]


def mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def encoded_to_raw_feature(encoded_feature: str, raw_features: list[str]) -> str:
    if encoded_feature in raw_features:
        return encoded_feature
    for feature in sorted(raw_features, key=len, reverse=True):
        if encoded_feature.startswith(f"{feature}_"):
            return feature
    return encoded_feature


def predict_full_model_dropout_probability(df: pd.DataFrame, metadata: dict) -> np.ndarray:
    full_features = metadata["ablation_feature_sets"]["full"]
    numeric = [f for f in metadata["feature_types"]["numeric_features"] if f in full_features]
    categorical = [f for f in metadata["feature_types"]["categorical_features"] if f in full_features]
    preprocessor = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)

    train_df = df[df["Split"] == "train"].copy()
    preprocessor.fit(train_df[full_features])
    x_all = preprocessor.transform_numpy(df[full_features])

    with open(FULL_MODEL_DIR / "full__lightgbm.pkl", "rb") as f:
        model = pickle.load(f)

    proba = model.predict_proba(x_all)
    classes = list(model.classes_)
    dropout_idx = classes.index(0)
    return proba[:, dropout_idx]


def fit_risk_matrix(df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, np.ndarray, TabularOneHotScaler]:
    features = [f for f in RISK_FEATURES if f in df.columns]
    risk_df = df[features].copy()
    risk_df["Predicted dropout probability"] = df["Predicted dropout probability"]

    numeric = [
        f
        for f in features
        if f in metadata["feature_types"]["numeric_features"]
    ] + ["Predicted dropout probability"]
    categorical = [
        f
        for f in features
        if f in metadata["feature_types"]["categorical_features"]
    ]
    preprocessor = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)
    matrix = preprocessor.fit_transform(risk_df).to_numpy(dtype=np.float32)
    return risk_df, matrix, preprocessor


def choose_k(matrix: np.ndarray, min_k: int = 3, max_k: int = 8) -> pd.DataFrame:
    rows = []
    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=50)
        labels = model.fit_predict(matrix)
        rows.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(matrix, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(matrix, labels)),
            }
        )
    return pd.DataFrame(rows)


def cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    target_counts = pd.crosstab(df["Cluster"], df["Target"], normalize="index")
    target_counts = target_counts.rename(columns={c: f"{c.lower()}_rate" for c in target_counts.columns})

    profile = df.groupby("Cluster").agg(
        n=("StudentID", "count"),
        predicted_dropout_probability_mean=("Predicted dropout probability", "mean"),
        age_mean=("Age at enrollment", "mean"),
        admission_grade_mean=("Admission grade", "mean"),
        first_sem_approved_mean=("Curricular units 1st sem (approved)", "mean"),
        first_sem_grade_mean=("Curricular units 1st sem (grade)", "mean"),
        second_sem_approved_mean=("Curricular units 2nd sem (approved)", "mean"),
        second_sem_grade_mean=("Curricular units 2nd sem (grade)", "mean"),
        tuition_not_up_to_date_rate=("Tuition fees up to date", lambda s: float((s == 0).mean())),
        debtor_rate=("Debtor", lambda s: float((s == 1).mean())),
        scholarship_rate=("Scholarship holder", lambda s: float((s == 1).mean())),
        displaced_rate=("Displaced", lambda s: float((s == 1).mean())),
    )
    profile = profile.join(target_counts).reset_index()
    if "dropout_rate" not in profile.columns:
        profile["dropout_rate"] = 0.0
    return profile.sort_values("dropout_rate", ascending=False)


def distinguishing_features(matrix: np.ndarray, labels: np.ndarray, columns: list[str], raw_features: list[str]) -> pd.DataFrame:
    overall = matrix.mean(axis=0)
    overall_std = matrix.std(axis=0)
    overall_std[overall_std == 0] = 1.0
    rows = []
    for cluster in sorted(np.unique(labels)):
        cluster_mean = matrix[labels == cluster].mean(axis=0)
        diff = (cluster_mean - overall) / overall_std
        encoded = pd.DataFrame(
            {
                "Cluster": cluster,
                "encoded_feature": columns,
                "feature": [encoded_to_raw_feature(col, raw_features) for col in columns],
                "standardized_difference": diff,
                "abs_standardized_difference": np.abs(diff),
            }
        )
        raw = (
            encoded.groupby(["Cluster", "feature"], as_index=False)
            .agg(
                standardized_difference=("standardized_difference", "mean"),
                max_abs_standardized_difference=("abs_standardized_difference", "max"),
                encoded_columns=("encoded_feature", "count"),
            )
            .sort_values(["Cluster", "max_abs_standardized_difference"], ascending=[True, False])
        )
        rows.append(raw)
    return pd.concat(rows, ignore_index=True)


def assign_policy_labels(profile: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in profile.iterrows():
        if row["dropout_rate"] >= 0.75 and row["tuition_not_up_to_date_rate"] >= 0.5:
            label = "Critical financial-academic risk"
            action = "Immediate tuition/payment support plus academic recovery plan"
        elif row["dropout_rate"] >= 0.6 and row["second_sem_approved_mean"] <= 2:
            label = "Severe academic non-progression risk"
            action = "Mandatory academic advising and course-completion intervention"
        elif row["dropout_rate"] >= 0.6 and row["tuition_not_up_to_date_rate"] >= 0.30:
            label = "Financial-academic emergency risk"
            action = "Combine financial outreach with urgent academic completion support"
        elif row["dropout_rate"] >= 0.5 and row["second_sem_approved_mean"] <= 4:
            label = "Academic delay with financial vulnerability"
            action = "Prioritize course-completion advising and payment-status monitoring"
        elif row["dropout_rate"] >= 0.45 and row["debtor_rate"] >= 0.25:
            label = "Financial vulnerability risk"
            action = "Financial aid review, debtor outreach, scholarship counseling"
        elif row["dropout_rate"] <= 0.25 and row["second_sem_approved_mean"] >= 8 and row["age_mean"] >= 26:
            label = "Older high-workload moderate risk"
            action = "Offer flexible scheduling and light-touch persistence monitoring"
        elif row["dropout_rate"] >= 0.35 and row["age_mean"] >= 28:
            label = "Older-student persistence risk"
            action = "Flexible scheduling, workload counseling, targeted retention contact"
        elif row["dropout_rate"] <= 0.20 and row["second_sem_approved_mean"] >= 5:
            label = "Low-risk academically stable group"
            action = "Maintain light monitoring; use as reference profile"
        else:
            label = "Moderate mixed risk"
            action = "Monitor academic progress and payment status"
        labels.append({"Cluster": row["Cluster"], "risk_cluster_label": label, "recommended_action": action})
    return profile.merge(pd.DataFrame(labels), on="Cluster", how="left")


def plot_cluster_dropout(profile: pd.DataFrame, plot_dir: Path = PLOT_DIR) -> None:
    plot_df = profile.sort_values("dropout_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["risk_cluster_label"], plot_df["dropout_rate"], color="#59A14F")
    ax.set_xlabel("Observed dropout rate")
    ax.set_title("Risk Clusters by Observed Dropout Rate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_dropout_rate.png", dpi=200)
    plt.close(fig)


def plot_pca(matrix: np.ndarray, labels: np.ndarray, df: pd.DataFrame, csv_dir: Path = CSV_DIR, plot_dir: Path = PLOT_DIR) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(matrix)
    pca_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Cluster": labels,
            "Target": df["Target"].to_numpy(),
        }
    )
    pca_df.to_csv(csv_dir / "cluster_pca_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"],
        cmap="viridis",
        s=18,
        alpha=0.75,
        linewidths=0,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    ax.set_title("Student Risk Clusters in PCA Space")
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(plot_dir / "student_risk_clusters_pca.png", dpi=200)
    plt.close(fig)


def plot_profile_heatmap(profile: pd.DataFrame, plot_dir: Path = PLOT_DIR) -> None:
    cols = [
        "dropout_rate",
        "predicted_dropout_probability_mean",
        "tuition_not_up_to_date_rate",
        "debtor_rate",
        "scholarship_rate",
        "age_mean",
        "first_sem_approved_mean",
        "second_sem_approved_mean",
        "second_sem_grade_mean",
    ]
    heat = profile.set_index("risk_cluster_label")[cols].copy()
    scaled = (heat - heat.mean()) / heat.std(ddof=0).replace(0, 1)

    fig, ax = plt.subplots(figsize=(11, 5.6))
    im = ax.imshow(scaled.to_numpy(), cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels(scaled.index)
    ax.set_title("Standardized Risk Cluster Profiles")
    fig.colorbar(im, ax=ax, label="Standardized value")
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_profile_heatmap.png", dpi=200)
    plt.close(fig)


def plot_k_selection(k_df: pd.DataFrame, plot_dir: Path = PLOT_DIR) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(k_df["k"], k_df["silhouette"], marker="o", color="#4C78A8", label="Silhouette")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("Silhouette score")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(k_df["k"], k_df["calinski_harabasz"], marker="s", color="#59A14F", label="Calinski-Harabasz")
    ax2.set_ylabel("Calinski-Harabasz")
    ax1.set_title("Cluster Number Selection")
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_number_selection.png", dpi=200)
    plt.close(fig)


def save_cluster_solution(
    df: pd.DataFrame,
    risk_df: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    preprocessor: TabularOneHotScaler,
    csv_dir: Path,
    plot_dir: Path,
    report_dir: Path,
    solution_name: str,
) -> pd.DataFrame:
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    working = df.copy()
    working["Cluster"] = labels
    clustered = working[
        [
            "StudentID",
            "Target",
            "Cluster",
            "Predicted dropout probability",
        ]
        + [f for f in RISK_FEATURES if f in working.columns]
    ].copy()

    profile = cluster_profiles(working)
    profile = assign_policy_labels(profile)
    distinguishing = distinguishing_features(
        matrix,
        labels,
        preprocessor.output_columns_,
        list(risk_df.drop(columns=["Cluster"], errors="ignore").columns),
    )

    clustered.to_csv(csv_dir / "student_risk_clusters.csv", index=False)
    profile.to_csv(csv_dir / "risk_cluster_profiles.csv", index=False)
    distinguishing.to_csv(csv_dir / "cluster_distinguishing_features_all.csv", index=False)
    distinguishing.groupby("Cluster").head(10).to_csv(csv_dir / "cluster_distinguishing_features_top10.csv", index=False)

    plot_cluster_dropout(profile, plot_dir)
    plot_pca(matrix, labels, working, csv_dir, plot_dir)
    plot_profile_heatmap(profile, plot_dir)

    high_risk = profile.sort_values("dropout_rate", ascending=False).iloc[0]
    low_risk = profile.sort_values("dropout_rate", ascending=True).iloc[0]
    report_lines = [
        f"# Risk Clustering Report: {solution_name}",
        "",
        "This solution clusters students using the whole dataset for practical intervention profiling.",
        "",
        f"The highest-risk cluster is **Cluster {int(high_risk['Cluster'])}**, labelled **{high_risk['risk_cluster_label']}**. It contains {int(high_risk['n'])} students, has observed dropout rate {high_risk['dropout_rate']:.1%}, mean predicted dropout probability {high_risk['predicted_dropout_probability_mean']:.1%}, tuition-not-up-to-date rate {high_risk['tuition_not_up_to_date_rate']:.1%}, debtor rate {high_risk['debtor_rate']:.1%}, and mean second-semester approved units {high_risk['second_sem_approved_mean']:.2f}. Recommended action: {high_risk['recommended_action']}.",
        "",
        f"The lowest-risk cluster is **Cluster {int(low_risk['Cluster'])}**, labelled **{low_risk['risk_cluster_label']}**. It contains {int(low_risk['n'])} students and has observed dropout rate {low_risk['dropout_rate']:.1%}, with mean second-semester approved units {low_risk['second_sem_approved_mean']:.2f} and mean second-semester grade {low_risk['second_sem_grade_mean']:.2f}.",
    ]
    (report_dir / "risk_clustering_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return profile


def main() -> None:
    mkdirs(CSV_DIR, PLOT_DIR, REPORT_DIR, MODEL_DIR)

    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")
    df["Predicted dropout probability"] = predict_full_model_dropout_probability(df, metadata)

    risk_df, matrix, preprocessor = fit_risk_matrix(df, metadata)
    with open(MODEL_DIR / "risk_clustering_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    k_df = choose_k(matrix)
    # Keep clusters practical for intervention design: choose best silhouette, but avoid fewer than 4 groups if close.
    best_silhouette = k_df["silhouette"].max()
    practical = k_df[k_df["silhouette"] >= best_silhouette - 0.015].copy()
    practical = practical[practical["k"] >= 4] if (practical["k"] >= 4).any() else practical
    selected_k = int(practical.sort_values(["silhouette", "calinski_harabasz"], ascending=[False, False]).iloc[0]["k"])

    model = KMeans(n_clusters=selected_k, random_state=RANDOM_STATE, n_init=100)
    labels = model.fit_predict(matrix)

    k_df.to_csv(CSV_DIR / "cluster_number_selection_metrics.csv", index=False)
    with open(MODEL_DIR / "risk_cluster_kmeans.pkl", "wb") as f:
        pickle.dump(model, f)

    plot_k_selection(k_df)
    profile = save_cluster_solution(
        df=df,
        risk_df=risk_df,
        matrix=matrix,
        labels=labels,
        preprocessor=preprocessor,
        csv_dir=CSV_DIR,
        plot_dir=PLOT_DIR,
        report_dir=REPORT_DIR,
        solution_name=f"Statistical k={selected_k}",
    )

    practical_k = 5
    practical_profile = None
    if practical_k != selected_k:
        practical_dir = OUT_DIR / "Practical_5_Clusters"
        practical_csv = practical_dir / "csv"
        practical_plots = practical_dir / "plots"
        practical_reports = practical_dir / "reports"
        practical_model = KMeans(n_clusters=practical_k, random_state=RANDOM_STATE, n_init=100)
        practical_labels = practical_model.fit_predict(matrix)
        with open(MODEL_DIR / "risk_cluster_kmeans_practical_5.pkl", "wb") as f:
            pickle.dump(practical_model, f)
        practical_profile = save_cluster_solution(
            df=df,
            risk_df=risk_df,
            matrix=matrix,
            labels=practical_labels,
            preprocessor=preprocessor,
            csv_dir=practical_csv,
            plot_dir=practical_plots,
            report_dir=practical_reports,
            solution_name="Practical k=5",
        )

    high_risk = profile.sort_values("dropout_rate", ascending=False).iloc[0]
    low_risk = profile.sort_values("dropout_rate", ascending=True).iloc[0]
    report_lines = [
        "# Risk Clustering Report",
        "",
        "This analysis clusters students using the whole dataset, not the train/validation/test split. The goal is not model evaluation, but practical intervention design. The clustering matrix combines key academic, financial, demographic, and course-related risk variables with the tuned full-feature model's predicted dropout probability.",
        "",
        f"The strongest statistical clustering solution uses **k = {selected_k}**. Selection considered silhouette and Calinski-Harabasz scores. Because intervention design may need more granular groups, a separate practical **k = 5** sensitivity solution is also saved under `Practical_5_Clusters/`.",
        "",
        "## Most Important Cluster Findings",
        "",
        f"The highest-risk cluster is **Cluster {int(high_risk['Cluster'])}**, labelled **{high_risk['risk_cluster_label']}**. It contains {int(high_risk['n'])} students, has observed dropout rate {high_risk['dropout_rate']:.1%}, mean predicted dropout probability {high_risk['predicted_dropout_probability_mean']:.1%}, tuition-not-up-to-date rate {high_risk['tuition_not_up_to_date_rate']:.1%}, debtor rate {high_risk['debtor_rate']:.1%}, and mean second-semester approved units {high_risk['second_sem_approved_mean']:.2f}. Recommended action: {high_risk['recommended_action']}.",
        "",
        f"The lowest-risk cluster is **Cluster {int(low_risk['Cluster'])}**, labelled **{low_risk['risk_cluster_label']}**. It contains {int(low_risk['n'])} students and has observed dropout rate {low_risk['dropout_rate']:.1%}, with mean second-semester approved units {low_risk['second_sem_approved_mean']:.2f} and mean second-semester grade {low_risk['second_sem_grade_mean']:.2f}.",
        "",
        "## Paper Use",
        "",
        "For the paper, use the k=3 solution if the goal is clean statistical separation, and use the k=5 practical sensitivity solution if the goal is richer intervention groups. The strongest figures are `cluster_dropout_rate.png` and `cluster_profile_heatmap.png`. These results translate the predictive model into actionable intervention groups rather than only reporting accuracy.",
    ]
    if practical_profile is not None:
        practical_summary = practical_profile.sort_values("dropout_rate", ascending=False)
        report_lines.extend(["", "## Practical k=5 Sensitivity Summary", ""])
        for _, row in practical_summary.iterrows():
            report_lines.append(
                f"- Cluster {int(row['Cluster'])}: {row['risk_cluster_label']}; n={int(row['n'])}; dropout={row['dropout_rate']:.1%}; predicted risk={row['predicted_dropout_probability_mean']:.1%}; second-semester approved={row['second_sem_approved_mean']:.2f}; tuition-not-up-to-date={row['tuition_not_up_to_date_rate']:.1%}."
            )
    (REPORT_DIR / "risk_clustering_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    status = {
        "rows_clustered": int(len(df)),
        "selected_k": selected_k,
        "risk_features": len(RISK_FEATURES),
        "encoded_features": int(matrix.shape[1]),
        "highest_dropout_cluster": int(high_risk["Cluster"]),
        "highest_dropout_rate": float(high_risk["dropout_rate"]),
    }
    (OUT_DIR / "stage8_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
