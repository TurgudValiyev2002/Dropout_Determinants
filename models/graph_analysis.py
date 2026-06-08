from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from models.preprocessing import TabularOneHotScaler

warnings.filterwarnings("ignore")


PREP_DIR = Path("results") / "2.Model Processing"
FULL_MODEL_DIR = Path("results") / "5.Full Feature Final Modeling" / "Hyp Tuned" / "models"
OUT_DIR = Path("results") / "9.Graph Analysis"
CSV_DIR = OUT_DIR / "csv"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"
MODEL_DIR = OUT_DIR / "models"
RANDOM_STATE = 123


STUDENT_GRAPH_FEATURES = [
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Age at enrollment",
    "Displaced",
    "Gender",
    "Course",
    "Admission grade",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
]


COURSE_PROFILE_FEATURES = [
    "dropout_rate",
    "graduate_rate",
    "enrolled_rate",
    "predicted_dropout_probability_mean",
    "tuition_not_up_to_date_rate",
    "debtor_rate",
    "scholarship_rate",
    "age_mean",
    "first_sem_approved_mean",
    "second_sem_approved_mean",
    "second_sem_grade_mean",
]


def mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


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
    dropout_idx = list(model.classes_).index(0)
    return proba[:, dropout_idx]


def build_student_matrix(df: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, np.ndarray, TabularOneHotScaler]:
    features = [f for f in STUDENT_GRAPH_FEATURES if f in df.columns]
    graph_df = df[features].copy()
    graph_df["Predicted dropout probability"] = df["Predicted dropout probability"]
    numeric = [f for f in features if f in metadata["feature_types"]["numeric_features"]] + ["Predicted dropout probability"]
    categorical = [f for f in features if f in metadata["feature_types"]["categorical_features"]]
    preprocessor = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)
    matrix = preprocessor.fit_transform(graph_df).to_numpy(dtype=np.float32)
    return graph_df, matrix, preprocessor


def build_knn_graph(matrix: np.ndarray, k: int = 12) -> nx.Graph:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            similarity = max(0.0, 1.0 - float(dist))
            if similarity > 0:
                if graph.has_edge(i, int(j)):
                    if similarity > graph[i][int(j)]["weight"]:
                        graph[i][int(j)]["weight"] = similarity
                else:
                    graph.add_edge(i, int(j), weight=similarity)
    return graph


def community_profiles(df: pd.DataFrame, community_col: str) -> pd.DataFrame:
    target = pd.crosstab(df[community_col], df["Target"], normalize="index")
    target = target.rename(columns={c: f"{c.lower()}_rate" for c in target.columns})
    profile = df.groupby(community_col).agg(
        n=("StudentID", "count"),
        predicted_dropout_probability_mean=("Predicted dropout probability", "mean"),
        age_mean=("Age at enrollment", "mean"),
        tuition_not_up_to_date_rate=("Tuition fees up to date", lambda s: float((s == 0).mean())),
        debtor_rate=("Debtor", lambda s: float((s == 1).mean())),
        scholarship_rate=("Scholarship holder", lambda s: float((s == 1).mean())),
        first_sem_approved_mean=("Curricular units 1st sem (approved)", "mean"),
        first_sem_grade_mean=("Curricular units 1st sem (grade)", "mean"),
        second_sem_approved_mean=("Curricular units 2nd sem (approved)", "mean"),
        second_sem_grade_mean=("Curricular units 2nd sem (grade)", "mean"),
    )
    profile = profile.join(target).reset_index()
    if "dropout_rate" not in profile.columns:
        profile["dropout_rate"] = 0.0
    return profile.sort_values("dropout_rate", ascending=False)


def label_student_communities(profile: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in profile.iterrows():
        if row["dropout_rate"] >= 0.75 and row["second_sem_approved_mean"] <= 1:
            label = "Collapsed academic progression community"
        elif row["dropout_rate"] >= 0.55 and row["tuition_not_up_to_date_rate"] >= 0.25:
            label = "Financial-academic risk community"
        elif row["dropout_rate"] >= 0.40 and row["second_sem_approved_mean"] <= 4:
            label = "Delayed progression community"
        elif row["dropout_rate"] <= 0.15 and row["second_sem_grade_mean"] >= 12:
            label = "Stable low-risk community"
        elif row["age_mean"] >= 27 and row["second_sem_approved_mean"] >= 7:
            label = "Older high-workload community"
        else:
            label = "Mixed-risk community"
        labels.append({"Student community": row["Student community"], "community_label": label})
    return profile.merge(pd.DataFrame(labels), on="Student community", how="left")


def graph_node_metrics(graph: nx.Graph, df: pd.DataFrame) -> pd.DataFrame:
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    pagerank = nx.pagerank(graph, weight="weight")
    metrics = pd.DataFrame(
        {
            "node": list(graph.nodes()),
            "StudentID": df["StudentID"].to_numpy(),
            "Target": df["Target"].to_numpy(),
            "Student community": df["Student community"].to_numpy(),
            "Predicted dropout probability": df["Predicted dropout probability"].to_numpy(),
            "degree": [degree[n] for n in graph.nodes()],
            "weighted_degree": [weighted_degree[n] for n in graph.nodes()],
            "pagerank": [pagerank[n] for n in graph.nodes()],
        }
    )
    return metrics.sort_values("pagerank", ascending=False)


def plot_student_graph(matrix: np.ndarray, df: pd.DataFrame, profile: pd.DataFrame) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(matrix)
    plot_df = pd.DataFrame(
        {
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Student community": df["Student community"],
            "Dropout probability": df["Predicted dropout probability"],
            "Target": df["Target"],
        }
    )
    plot_df.to_csv(CSV_DIR / "student_graph_pca_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        plot_df["PC1"],
        plot_df["PC2"],
        c=plot_df["Dropout probability"],
        cmap="magma_r",
        s=16,
        alpha=0.72,
        linewidths=0,
    )
    ax.set_title("Student Risk Similarity Graph Communities")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    fig.colorbar(scatter, ax=ax, label="Predicted dropout probability")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "student_similarity_graph_pca_risk.png", dpi=220)
    plt.close(fig)

    top = profile.sort_values("dropout_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top["community_label"], top["dropout_rate"], color="#59A14F")
    ax.set_xlabel("Observed dropout rate")
    ax.set_title("Student Graph Communities by Dropout Rate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "student_community_dropout_rates.png", dpi=220)
    plt.close(fig)


def course_profiles(df: pd.DataFrame) -> pd.DataFrame:
    target = pd.crosstab(df["Course"], df["Target"], normalize="index")
    target = target.rename(columns={c: f"{c.lower()}_rate" for c in target.columns})
    profile = df.groupby("Course").agg(
        n=("StudentID", "count"),
        predicted_dropout_probability_mean=("Predicted dropout probability", "mean"),
        age_mean=("Age at enrollment", "mean"),
        tuition_not_up_to_date_rate=("Tuition fees up to date", lambda s: float((s == 0).mean())),
        debtor_rate=("Debtor", lambda s: float((s == 1).mean())),
        scholarship_rate=("Scholarship holder", lambda s: float((s == 1).mean())),
        first_sem_approved_mean=("Curricular units 1st sem (approved)", "mean"),
        second_sem_approved_mean=("Curricular units 2nd sem (approved)", "mean"),
        second_sem_grade_mean=("Curricular units 2nd sem (grade)", "mean"),
    )
    profile = profile.join(target).reset_index()
    if "dropout_rate" not in profile.columns:
        profile["dropout_rate"] = 0.0
    return profile


def build_course_graph(profile: pd.DataFrame, threshold: float = 0.90, top_k: int | None = None) -> tuple[nx.Graph, pd.DataFrame]:
    features = [c for c in COURSE_PROFILE_FEATURES if c in profile.columns]
    x = profile[features].copy()
    x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
    sim = cosine_similarity(x)
    graph = nx.Graph()
    for _, row in profile.iterrows():
        graph.add_node(int(row["Course"]), n=int(row["n"]), dropout_rate=float(row["dropout_rate"]))
    edges = []
    courses = profile["Course"].astype(int).tolist()
    candidate_edges = []
    for i, course_i in enumerate(courses):
        for j in range(i + 1, len(courses)):
            candidate_edges.append((course_i, courses[j], float(sim[i, j])))
    if top_k is None:
        selected = [edge for edge in candidate_edges if edge[2] >= threshold]
    else:
        selected_pairs = set()
        for course in courses:
            incident = [edge for edge in candidate_edges if edge[0] == course or edge[1] == course]
            incident = sorted(incident, key=lambda edge: edge[2], reverse=True)[:top_k]
            for a, b, similarity in incident:
                selected_pairs.add((min(a, b), max(a, b), similarity))
        selected = sorted(selected_pairs, key=lambda edge: edge[2], reverse=True)
    for course_i, course_j, similarity in selected:
        graph.add_edge(course_i, course_j, weight=similarity)
        edges.append({"source_course": course_i, "target_course": course_j, "similarity": similarity})
    return graph, pd.DataFrame(edges).sort_values("similarity", ascending=False)


def course_community_profiles(profile: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    if graph.number_of_edges() == 0:
        profile["Course community"] = np.arange(len(profile))
    else:
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")
        mapping = {}
        for idx, community in enumerate(communities):
            for node in community:
                mapping[node] = idx
        profile["Course community"] = profile["Course"].astype(int).map(mapping)
    comm = profile.groupby("Course community").agg(
        courses=("Course", lambda s: "; ".join(map(str, sorted(s.astype(int).tolist())))),
        n_courses=("Course", "count"),
        total_students=("n", "sum"),
        dropout_rate_mean=("dropout_rate", "mean"),
        predicted_dropout_probability_mean=("predicted_dropout_probability_mean", "mean"),
        tuition_not_up_to_date_rate_mean=("tuition_not_up_to_date_rate", "mean"),
        second_sem_approved_mean=("second_sem_approved_mean", "mean"),
        second_sem_grade_mean=("second_sem_grade_mean", "mean"),
    ).reset_index()
    return comm.sort_values("dropout_rate_mean", ascending=False), profile


def plot_course_graph(profile: pd.DataFrame, graph: nx.Graph, comm_profile: pd.DataFrame, suffix: str = "") -> None:
    suffix_part = f"_{suffix}" if suffix else ""
    fig, ax = plt.subplots(figsize=(10, 7))
    if graph.number_of_edges() > 0:
        pos = nx.spring_layout(graph, seed=RANDOM_STATE, weight="weight")
        node_colors = [graph.nodes[n]["dropout_rate"] for n in graph.nodes()]
        node_sizes = [max(120, graph.nodes[n]["n"] * 2.5) for n in graph.nodes()]
        nx.draw_networkx_edges(graph, pos, ax=ax, alpha=0.25, width=1.2)
        nodes = nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap="magma_r",
            ax=ax,
            alpha=0.92,
        )
        nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
        fig.colorbar(nodes, ax=ax, label="Course dropout rate")
    ax.set_title("Course-Risk Similarity Graph" if not suffix else f"Course-Risk Similarity Graph ({suffix.replace('_', ' ').title()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"course_risk_similarity_graph{suffix_part}.png", dpi=220)
    plt.close(fig)

    plot_df = profile.sort_values("dropout_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_df["Course"].astype(str), plot_df["dropout_rate"], color="#4C78A8")
    ax.set_xlabel("Observed dropout rate")
    ax.set_ylabel("Course")
    ax.set_title("Dropout Rate by Course")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"course_dropout_rates{suffix_part}.png", dpi=220)
    plt.close(fig)


def plot_course_graph_interpretable(profile: pd.DataFrame, graph: nx.Graph) -> None:
    """Create a publication-oriented course graph with dropout labels and risk emphasis."""
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=RANDOM_STATE, weight="weight", k=1.15)

    dropout = {int(row["Course"]): float(row["dropout_rate"]) for _, row in profile.iterrows()}
    sizes = {int(row["Course"]): int(row["n"]) for _, row in profile.iterrows()}
    predicted = {
        int(row["Course"]): float(row["predicted_dropout_probability_mean"])
        for _, row in profile.iterrows()
    }

    edge_widths = [
        0.8 + 5.0 * max(0.0, graph[u][v].get("weight", 0.0) - 0.65)
        for u, v in graph.edges()
    ]
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        edge_color="#9E9E9E",
        alpha=0.42,
    )

    node_colors = [dropout[n] for n in graph.nodes()]
    node_sizes = [260 + sizes[n] * 4.2 for n in graph.nodes()]
    node_edge_colors = [
        "#B22222" if dropout[n] >= 0.45 else "#2E7D32" if dropout[n] <= 0.25 else "#555555"
        for n in graph.nodes()
    ]
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap="magma_r",
        edgecolors=node_edge_colors,
        linewidths=2.2,
        alpha=0.95,
    )

    labels = {
        n: f"{n}\nD={dropout[n] * 100:.0f}%\nn={sizes[n]}"
        for n in graph.nodes()
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8.5, font_weight="bold", ax=ax)

    high_risk_courses = sorted([n for n in graph.nodes() if dropout[n] >= 0.45])
    low_risk_courses = sorted([n for n in graph.nodes() if dropout[n] <= 0.25])
    subtitle = (
        f"High-risk courses: {', '.join(map(str, high_risk_courses))} | "
        f"Low-risk courses: {', '.join(map(str, low_risk_courses))}"
    )
    ax.set_title(
        "Interpretable Course-Risk Similarity Graph\n"
        "Node size = students, color = dropout rate, border = risk tier, edge width = similarity",
        fontsize=13,
        pad=16,
    )
    ax.text(
        0.5,
        -0.04,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.axis("off")
    cbar = fig.colorbar(nodes, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Observed course dropout rate")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "course_risk_similarity_graph_relaxed_top3_interpretable.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    interpretation_rows = []
    for n in sorted(graph.nodes(), key=lambda node: dropout[node], reverse=True):
        neighbors = sorted(
            graph.neighbors(n),
            key=lambda nb: graph[n][nb].get("weight", 0.0),
            reverse=True,
        )
        interpretation_rows.append(
            {
                "Course": n,
                "n": sizes[n],
                "dropout_rate": dropout[n],
                "predicted_dropout_probability_mean": predicted[n],
                "risk_tier": "high" if dropout[n] >= 0.45 else "low" if dropout[n] <= 0.25 else "moderate",
                "nearest_similar_courses": "; ".join(map(str, neighbors[:3])),
                "mean_neighbor_dropout_rate": float(np.mean([dropout[nb] for nb in neighbors])) if neighbors else np.nan,
            }
        )
    pd.DataFrame(interpretation_rows).to_csv(
        CSV_DIR / "course_risk_interpretable_graph_node_summary.csv",
        index=False,
    )


def main() -> None:
    mkdirs(CSV_DIR, PLOT_DIR, REPORT_DIR, MODEL_DIR)
    metadata = json.loads((PREP_DIR / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(PREP_DIR / "csv" / "modeling_master_dataset_with_splits.csv")
    df["Predicted dropout probability"] = predict_full_model_dropout_probability(df, metadata)

    graph_df, matrix, preprocessor = build_student_matrix(df, metadata)
    with open(MODEL_DIR / "student_graph_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    student_graph = build_knn_graph(matrix, k=12)
    communities = nx.community.greedy_modularity_communities(student_graph, weight="weight")
    community_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            community_map[node] = idx
    df["Student community"] = [community_map[i] for i in range(len(df))]
    student_profile = community_profiles(df, "Student community")
    student_profile = label_student_communities(student_profile)
    node_metrics = graph_node_metrics(student_graph, df)

    student_profile.to_csv(CSV_DIR / "student_graph_community_profiles.csv", index=False)
    node_metrics.to_csv(CSV_DIR / "student_graph_node_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "nodes": student_graph.number_of_nodes(),
                "edges": student_graph.number_of_edges(),
                "density": nx.density(student_graph),
                "communities": len(communities),
                "modularity": nx.community.modularity(student_graph, communities, weight="weight"),
            }
        ]
    ).to_csv(CSV_DIR / "student_graph_metrics.csv", index=False)
    nx.write_gexf(student_graph, MODEL_DIR / "student_similarity_graph.gexf")
    plot_student_graph(matrix, df, student_profile)

    c_profile = course_profiles(df)
    course_graph, course_edges = build_course_graph(c_profile, threshold=0.90)
    course_comm, c_profile = course_community_profiles(c_profile, course_graph)
    course_graph_relaxed, course_edges_relaxed = build_course_graph(c_profile.drop(columns=["Course community"], errors="ignore"), top_k=3)
    course_comm_relaxed, c_profile_relaxed = course_community_profiles(c_profile.drop(columns=["Course community"], errors="ignore"), course_graph_relaxed)
    c_profile.to_csv(CSV_DIR / "course_risk_profiles.csv", index=False)
    course_edges.to_csv(CSV_DIR / "course_risk_similarity_edges.csv", index=False)
    course_comm.to_csv(CSV_DIR / "course_risk_community_profiles.csv", index=False)
    c_profile_relaxed.to_csv(CSV_DIR / "course_risk_profiles_relaxed_top3.csv", index=False)
    course_edges_relaxed.to_csv(CSV_DIR / "course_risk_similarity_edges_relaxed_top3.csv", index=False)
    course_comm_relaxed.to_csv(CSV_DIR / "course_risk_community_profiles_relaxed_top3.csv", index=False)
    pd.DataFrame(
        [
            {
                "nodes": course_graph.number_of_nodes(),
                "edges": course_graph.number_of_edges(),
                "density": nx.density(course_graph),
                "communities": int(c_profile["Course community"].nunique()),
            }
        ]
    ).to_csv(CSV_DIR / "course_graph_metrics.csv", index=False)
    nx.write_gexf(course_graph, MODEL_DIR / "course_risk_similarity_graph.gexf")
    nx.write_gexf(course_graph_relaxed, MODEL_DIR / "course_risk_similarity_graph_relaxed_top3.gexf")
    plot_course_graph(c_profile, course_graph, course_comm)
    plot_course_graph(c_profile_relaxed, course_graph_relaxed, course_comm_relaxed, suffix="relaxed_top3")
    plot_course_graph_interpretable(c_profile_relaxed, course_graph_relaxed)

    high_student = student_profile.sort_values("dropout_rate", ascending=False).iloc[0]
    high_course = c_profile.sort_values("dropout_rate", ascending=False).iloc[0]
    high_course_comm = course_comm.sort_values("dropout_rate_mean", ascending=False).iloc[0]
    high_course_comm_relaxed = course_comm_relaxed.sort_values("dropout_rate_mean", ascending=False).iloc[0]

    report = [
        "# Graph Analysis Report",
        "",
        "This analysis builds two graphs using the whole dataset. The student-risk graph connects students with similar academic, financial, demographic, course, and predicted-risk profiles. The course-risk graph connects courses that have similar dropout-risk profiles.",
        "",
        "## Student-Risk Similarity Graph",
        "",
        f"The student graph contains {student_graph.number_of_nodes()} nodes and {student_graph.number_of_edges()} weighted similarity edges. Greedy modularity community detection found {len(communities)} communities with modularity {nx.community.modularity(student_graph, communities, weight='weight'):.3f}.",
        "",
        f"The highest-risk student community is **Community {int(high_student['Student community'])}**, labelled **{high_student['community_label']}**. It contains {int(high_student['n'])} students, has dropout rate {high_student['dropout_rate']:.1%}, mean predicted dropout probability {high_student['predicted_dropout_probability_mean']:.1%}, tuition-not-up-to-date rate {high_student['tuition_not_up_to_date_rate']:.1%}, and mean second-semester approved units {high_student['second_sem_approved_mean']:.2f}.",
        "",
        "## Course-Risk Similarity Graph",
        "",
        f"The highest-dropout course profile is Course {int(high_course['Course'])}, with {int(high_course['n'])} students, dropout rate {high_course['dropout_rate']:.1%}, and mean predicted dropout probability {high_course['predicted_dropout_probability_mean']:.1%}.",
        "",
        f"The highest-risk course community contains courses {high_course_comm['courses']}, with mean dropout rate {high_course_comm['dropout_rate_mean']:.1%}. This suggests that course-level risk may cluster across programs with similar academic progression and financial-risk profiles rather than appearing only in isolated courses.",
        "",
        f"Because the strict 0.90 similarity graph is intentionally conservative, a relaxed top-3-neighbor course graph is also saved. Its highest-risk course community contains courses {high_course_comm_relaxed['courses']}, with mean dropout rate {high_course_comm_relaxed['dropout_rate_mean']:.1%}. This relaxed view is more useful for visualization and for finding groups of courses with similar risk profiles.",
        "",
        "## Paper Use",
        "",
        "Recommended figures: `student_similarity_graph_pca_risk.png`, `student_community_dropout_rates.png`, `course_risk_similarity_graph_relaxed_top3_interpretable.png`, and `course_dropout_rates.png`. Recommended tables: `student_graph_community_profiles.csv`, `course_risk_profiles.csv`, `course_risk_community_profiles_relaxed_top3.csv`, and `course_risk_interpretable_graph_node_summary.csv`.",
    ]
    (REPORT_DIR / "graph_analysis_report.md").write_text("\n".join(report), encoding="utf-8")

    status = {
        "student_nodes": student_graph.number_of_nodes(),
        "student_edges": student_graph.number_of_edges(),
        "student_communities": len(communities),
        "course_nodes": course_graph.number_of_nodes(),
        "course_edges": course_graph.number_of_edges(),
        "course_communities": int(c_profile["Course community"].nunique()),
        "course_edges_relaxed_top3": course_graph_relaxed.number_of_edges(),
        "course_communities_relaxed_top3": int(c_profile_relaxed["Course community"].nunique()),
    }
    (OUT_DIR / "stage9_run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
