from __future__ import annotations

import json
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    silhouette_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold


ROOT = Path(__file__).resolve().parent / "Dropout_Determinants"
sys.path.insert(0, str(ROOT))
from models.preprocessing import TabularOneHotScaler  # noqa: E402

OUT = Path(__file__).resolve().parent / "critical_results"
OUT.mkdir(parents=True, exist_ok=True)
PREP = ROOT / "results" / "2.Model Processing"
META = json.loads((PREP / "metadata" / "feature_sets_metadata.json").read_text(encoding="utf-8"))
DF = pd.read_csv(PREP / "csv" / "modeling_master_dataset_with_splits.csv")
TARGET = META["targets"]["multiclass"]
CLASS_NAMES = sorted(DF[TARGET].unique().tolist())
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
DF["y"] = DF[TARGET].map(CLASS_TO_ID).astype(int)
RNG = np.random.default_rng(123)


SELECTED_PARAMS = {
    "A_pre_enrollment": {"learning_rate": 0.03, "max_depth": 5, "n_estimators": 400, "num_leaves": 15},
    "B_first_semester": {"learning_rate": 0.03, "max_depth": -1, "n_estimators": 200, "num_leaves": 31},
    "C_second_semester": {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 200, "num_leaves": 31},
}
MODEL_PATHS = {
    stage: ROOT / "results" / "4.Hyperparameter Tuning" / "models" / f"{stage}__lightgbm.pkl"
    for stage in SELECTED_PARAMS
}


def prep(train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    numeric = [f for f in META["feature_types"]["numeric_features"] if f in features]
    categorical = [f for f in META["feature_types"]["categorical_features"] if f in features]
    p = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)
    return p.fit_transform(train[features]).to_numpy(), p.transform_numpy(test[features])


def model(params: dict, seed: int = 123) -> LGBMClassifier:
    return LGBMClassifier(
        **params,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
        subsample=0.9,
        colsample_bytree=0.9,
    )


def metric_row(y, pred, proba):
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "log_loss": log_loss(y, proba, labels=[0, 1, 2]),
    }


def percentile_ci(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def selected_test_analysis():
    test = DF[DF["Split"] == "test"].copy()
    train = DF[DF["Split"] == "train"].copy()
    y = test["y"].to_numpy()
    predictions = {}
    class_rows = []
    confusion_rows = []
    summary = []
    for stage in SELECTED_PARAMS:
        features = META["stage_feature_sets"][stage]
        xtr, xt = prep(train, test, features)
        # Refit from the recorded parameters. Loading the repository pickle under
        # a different scikit-learn version altered its label encoding.
        fitted = model(SELECTED_PARAMS[stage])
        fitted.fit(xtr, train["y"])
        pred = fitted.predict(xt).astype(int)
        proba = fitted.predict_proba(xt)
        predictions[stage] = (pred, proba)
        report = classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        for name in CLASS_NAMES:
            class_rows.append({"stage": stage, "class": name, **{k: report[name][k] for k in ("precision", "recall", "f1-score", "support")}})
        cm = confusion_matrix(y, pred, labels=[0, 1, 2])
        for i, actual in enumerate(CLASS_NAMES):
            for j, predicted in enumerate(CLASS_NAMES):
                confusion_rows.append({"stage": stage, "actual": actual, "predicted": predicted, "n": int(cm[i, j])})
        values = {k: [] for k in ("macro_f1", "balanced_accuracy", "log_loss")}
        for _ in range(3000):
            idx = RNG.integers(0, len(y), len(y))
            m = metric_row(y[idx], pred[idx], proba[idx])
            for k, v in m.items():
                values[k].append(v)
        point = metric_row(y, pred, proba)
        row = {"stage": stage, **point}
        for k, vals in values.items():
            lo, hi = percentile_ci(vals)
            row[f"{k}_ci_low"] = lo
            row[f"{k}_ci_high"] = hi
        summary.append(row)
    pd.DataFrame(class_rows).to_csv(OUT / "selected_class_metrics.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(OUT / "selected_confusion_matrices.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "selected_test_bootstrap_ci.csv", index=False)

    diff_rows = []
    for earlier, later in [("A_pre_enrollment", "B_first_semester"), ("B_first_semester", "C_second_semester")]:
        diffs = {k: [] for k in ("macro_f1", "balanced_accuracy", "log_loss")}
        ep, eprob = predictions[earlier]
        lp, lprob = predictions[later]
        for _ in range(5000):
            idx = RNG.integers(0, len(y), len(y))
            em = metric_row(y[idx], ep[idx], eprob[idx])
            lm = metric_row(y[idx], lp[idx], lprob[idx])
            for k in diffs:
                diffs[k].append(lm[k] - em[k])
        for k, vals in diffs.items():
            lo, hi = percentile_ci(vals)
            diff_rows.append({"comparison": f"{later} minus {earlier}", "metric": k, "difference": metric_row(y, lp, lprob)[k] - metric_row(y, ep, eprob)[k], "ci_low": lo, "ci_high": hi})
    pd.DataFrame(diff_rows).to_csv(OUT / "paired_stage_bootstrap_differences.csv", index=False)


def repeated_cv_and_baselines():
    dev = DF[DF["Split"].isin(["train", "validation"])].reset_index(drop=True)
    test = DF[DF["Split"] == "test"].copy()
    y = dev["y"].to_numpy()
    rows = []
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)
    for stage, params in SELECTED_PARAMS.items():
        features = META["stage_feature_sets"][stage]
        for fold, (tr, va) in enumerate(splitter.split(dev, y), start=1):
            train_fold, val_fold = dev.iloc[tr], dev.iloc[va]
            xtr, xva = prep(train_fold, val_fold, features)
            fitted = model(params, seed=123 + fold)
            fitted.fit(xtr, train_fold["y"])
            pred, proba = fitted.predict(xva), fitted.predict_proba(xva)
            rows.append({"stage": stage, "fold": fold, **metric_row(val_fold["y"].to_numpy(), pred, proba)})
    cv = pd.DataFrame(rows)
    cv.to_csv(OUT / "repeated_5x5_cv_folds.csv", index=False)
    cv.groupby("stage")[["macro_f1", "balanced_accuracy", "log_loss"]].agg(["mean", "std", "min", "max"]).to_csv(OUT / "repeated_5x5_cv_summary.csv")

    baseline_rows = []
    baseline_train = DF[DF["Split"] == "train"].copy()
    for stage in SELECTED_PARAMS:
        features = META["stage_feature_sets"][stage]
        xdev, xtest = prep(baseline_train, test, features)
        for name, fitted in [
            ("Most-frequent dummy", DummyClassifier(strategy="most_frequent")),
            ("Multinomial logistic regression", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=123)),
        ]:
            fitted.fit(xdev, baseline_train["y"])
            pred, proba = fitted.predict(xtest), fitted.predict_proba(xtest)
            baseline_rows.append({"stage": stage, "model": name, **metric_row(test["y"].to_numpy(), pred, proba)})
    pd.DataFrame(baseline_rows).to_csv(OUT / "transparent_baselines_test.csv", index=False)


def fixed_model_ablation():
    train = DF[DF["Split"] == "train"].copy()
    test = DF[DF["Split"] == "test"].copy()
    params = {"learning_rate": 0.05, "max_depth": 5, "n_estimators": 200, "num_leaves": 31}
    labels = {
        "academic_only": "Academic only",
        "academic_plus_financial_demographic": "Academic + demographic-financial",
        "academic_plus_parental": "Academic + parental",
        "academic_plus_macro": "Academic + macroeconomic",
        "academic_plus_course_application": "Academic + course-application",
        "full": "Full feature",
    }
    rows = []
    y = test["y"].to_numpy()
    for key, label in labels.items():
        features = META["ablation_feature_sets"][key]
        xtr, xt = prep(train, test, features)
        fitted = model(params)
        fitted.fit(xtr, train["y"])
        pred, proba = fitted.predict(xt), fitted.predict_proba(xt)
        boot = []
        for _ in range(2000):
            idx = RNG.integers(0, len(y), len(y))
            boot.append(f1_score(y[idx], pred[idx], average="macro"))
        lo, hi = percentile_ci(boot)
        rows.append({"feature_set": label, "raw_features": len(features), **metric_row(y, pred, proba), "macro_f1_ci_low": lo, "macro_f1_ci_high": hi})
    pd.DataFrame(rows).to_csv(OUT / "fixed_lightgbm_ablation_test.csv", index=False)


RISK_FEATURES = [
    "Tuition fees up to date", "Debtor", "Scholarship holder", "Age at enrollment", "Displaced", "Gender", "Course", "Application mode",
    "Admission grade", "Previous qualification (grade)", "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
]


def unsupervised_clustering():
    # Outcome and model-predicted risk are intentionally excluded.
    x, _ = prep(DF, DF, RISK_FEATURES)
    selection = []
    labels_by_k = {}
    for k in range(3, 9):
        fitted = KMeans(n_clusters=k, random_state=123, n_init=50).fit(x)
        labels_by_k[k] = fitted.labels_
        selection.append({"k": k, "silhouette": silhouette_score(x, fitted.labels_), "calinski_harabasz": calinski_harabasz_score(x, fitted.labels_), "inertia": fitted.inertia_})
    pd.DataFrame(selection).to_csv(OUT / "unsupervised_cluster_selection.csv", index=False)
    k = max(selection, key=lambda r: r["silhouette"])["k"]
    seed_labels = [KMeans(n_clusters=k, random_state=s, n_init=20).fit_predict(x) for s in range(30)]
    aris = [adjusted_rand_score(a, b) for a, b in combinations(seed_labels, 2)]
    labels = KMeans(n_clusters=k, random_state=123, n_init=100).fit_predict(x)
    work = DF.copy()
    work["cluster"] = labels
    target_rates = pd.crosstab(work["cluster"], work[TARGET], normalize="index")
    profiles = work.groupby("cluster").agg(
        n=("StudentID", "count"), age_mean=("Age at enrollment", "mean"),
        first_approved_mean=("Curricular units 1st sem (approved)", "mean"),
        first_grade_mean=("Curricular units 1st sem (grade)", "mean"),
        second_approved_mean=("Curricular units 2nd sem (approved)", "mean"),
        second_grade_mean=("Curricular units 2nd sem (grade)", "mean"),
        tuition_not_current_rate=("Tuition fees up to date", lambda s: (s == 0).mean()),
        debtor_rate=("Debtor", lambda s: (s == 1).mean()),
    ).join(target_rates).reset_index()
    profiles.to_csv(OUT / "unsupervised_cluster_profiles.csv", index=False)
    (OUT / "unsupervised_cluster_stability.json").write_text(json.dumps({"selected_k": int(k), "mean_pairwise_ari": float(np.mean(aris)), "min_pairwise_ari": float(np.min(aris)), "max_pairwise_ari": float(np.max(aris)), "n_seeds": 30}, indent=2), encoding="utf-8")


def environment_and_search_budget():
    candidates = pd.read_csv(ROOT / "results" / "4.Hyperparameter Tuning" / "csv" / "tuning_candidate_validation_results.csv")
    budget = candidates.groupby(["stage", "model"]).size().reset_index(name="candidate_configurations")
    budget.to_csv(OUT / "tuning_search_budget.csv", index=False)
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "cpu_count": os.cpu_count(),
        "random_seed": 123,
        "note": "Robustness analyses were rerun on 31 August 2026; original repository model artifacts may have been created under earlier package versions.",
    }
    (OUT / "analysis_environment.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    selected_test_analysis()
    repeated_cv_and_baselines()
    fixed_model_ablation()
    unsupervised_clustering()
    environment_and_search_budget()
    print(OUT)
