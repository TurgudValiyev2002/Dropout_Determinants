"""Export verified evidence, not LLM explanations. Source files are read-only.

Run with Python -B. Fit only on the original training split; preserve the
validation/test split. Never use evaluation/ files as LLM prompt inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

PARAMS = {
    "A_pre_enrollment": dict(learning_rate=0.03, max_depth=5, n_estimators=400, num_leaves=15),
    "B_first_semester": dict(learning_rate=0.03, max_depth=-1, n_estimators=200, num_leaves=31),
    "C_second_semester": dict(learning_rate=0.05, max_depth=5, n_estimators=200, num_leaves=31),
}
TIMING = {
    "A_pre_enrollment": "Pre-enrollment feature specification",
    "B_first_semester": "First-semester feature specification",
    "C_second_semester": "Second-semester feature specification",
}
NOTES = [
    "Explain this model prediction, not the student's actual outcome.",
    "Model probabilities are uncalibrated estimates, not certainties.",
    "SHAP contributions are in the Dropout class raw-score units, not percentage points.",
    "A positive contribution raises the Dropout raw score relative to its SHAP baseline; a negative contribution lowers it.",
    "Contributions are model associations, not causal effects or proven intervention benefits.",
    "Categorical values are original dataset codes. Do not invent meanings without a verified codebook.",
    "Financial variables may not have been recorded at the nominal prediction time; stage names do not establish prospective availability.",
    "Do not infer motives, sensitive characteristics beyond recorded values, or unobserved circumstances.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Extracted Supplementary Analysis S1 root")
    parser.add_argument("--output", required=True, type=Path, help="A NEW output directory; existing paths are refused")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(source.is_dir(), "Source directory does not exist")
    require(not output.exists(), "Output already exists. Choose a new versioned output path; nothing was overwritten.")
    require(not output.is_relative_to(source), "Output must be outside the read-only source directory")
    require(args.n_jobs > 0, "n-jobs must be positive")
    root = source / "Dropout_Determinants"
    data_path = root / "results/2.Model Processing/csv/modeling_master_dataset_with_splits.csv"
    metadata_path = root / "results/2.Model Processing/metadata/feature_sets_metadata.json"
    reference_path = source / "consistency_07_10/selected_test_bootstrap_ci.csv"
    tracked = [data_path, metadata_path, reference_path, root / "models/preprocessing.py", Path(__file__).resolve()]
    before = {str(p): digest(p) for p in tracked}
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    data = pd.read_csv(data_path)
    target = meta["targets"]["multiclass"]
    classes = sorted(data[target].unique().tolist())
    require(classes == ["Dropout", "Enrolled", "Graduate"], "Unexpected class labels")
    require(data.StudentID.is_unique and data.StudentID.notna().all(), "Invalid student identifiers")
    counts = data.Split.value_counts().to_dict()
    require(counts == {"train": 3097, "validation": 885, "test": 442}, f"Unexpected split counts: {counts}")
    require(set(meta["stage_feature_sets"]) == set(PARAMS), "Unexpected feature specifications")
    train = data[data.Split == "train"]
    y_train = train[target].map(dict(zip(classes, range(3)))).to_numpy()
    reference = pd.read_csv(reference_path).set_index("stage")
    # Import only the pure preprocessing helper, never critical_analysis.py,
    # which creates an output directory as an import side effect.
    sys.path.insert(0, str(root))
    from models.preprocessing import TabularOneHotScaler

    cases = {split: [] for split in ("validation", "test")}
    truth, checks, fitted_models = [], [], []
    forbidden = set(meta["targets"].values()) | {"StudentID", "Split", "y"}
    for stage, params in PARAMS.items():
        print(f"Verifying {stage} ...", flush=True)
        features = meta["stage_feature_sets"][stage]
        require(len(features) == len(set(features)) and not (set(features) & forbidden), "Unsafe feature list")
        numeric = [f for f in meta["feature_types"]["numeric_features"] if f in features]
        categorical = [f for f in meta["feature_types"]["categorical_features"] if f in features]
        require(set(numeric).isdisjoint(categorical) and set(numeric + categorical) == set(features), "Incomplete feature typing")
        prep = TabularOneHotScaler(numeric_features=numeric, categorical_features=categorical)
        x_train = prep.fit_transform(train[features])
        # Exact mapping of every transformed column back to one original variable.
        owners = list(numeric)
        for feature in categorical:
            owners.extend([feature] * len(prep.categories_[feature]))
        require(len(owners) == x_train.shape[1], "Transformed feature mapping mismatch")
        model = LGBMClassifier(**params, random_state=123, class_weight="balanced", n_jobs=args.n_jobs,
                               verbose=-1, subsample=0.9, colsample_bytree=0.9)
        model.fit(x_train.to_numpy(), y_train)
        require(np.array_equal(model.classes_, np.arange(3)), "Unexpected model class order")
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent", model_output="raw")
        baseline = np.asarray(explainer.expected_value, dtype=float)
        require(baseline.shape == (3,), "Unexpected SHAP baseline shape")
        for split in cases:
            rows = data[data.Split == split]
            x = prep.transform_numpy(rows[features])
            probabilities = model.predict_proba(x)
            raw = model.booster_.predict(x, raw_score=True)
            predictions = np.argmax(probabilities, axis=1)
            values = explainer.shap_values(x)
            values = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
            require(values.shape == (len(rows), x.shape[1], 3), f"Unexpected SHAP shape: {values.shape}")
            require(np.isfinite(values).all() and np.isfinite(probabilities).all(), "Nonfinite results")
            require(np.allclose(probabilities.sum(axis=1), 1, atol=1e-10), "Probabilities do not sum to one")
            shifted = raw - raw.max(axis=1, keepdims=True)
            softmax = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            require(np.allclose(softmax, probabilities, atol=1e-10), "Raw scores/probabilities mismatch")
            error = float(np.max(np.abs(values.sum(axis=1) + baseline - raw)))
            require(error < 1e-6, f"SHAP additivity failed: {error}")
            grouped = np.column_stack([values[:, np.array(owners) == f, 0].sum(axis=1) for f in features])
            require(np.allclose(grouped.sum(axis=1) + baseline[0], raw[:, 0], atol=1e-6), "Grouped SHAP failed")
            actual = rows[target].map(dict(zip(classes, range(3)))).to_numpy()
            macro_f1 = float(f1_score(actual, predictions, average="macro"))
            aggregate_error = None
            if split == "test":
                require(abs(macro_f1 - reference.loc[stage, "macro_f1"]) < 1e-8, "Reference model F1 mismatch")
                shap_path = source / f"consistency_07_10/shap_{stage}.csv"
                tracked.append(shap_path)
                before[str(shap_path)] = digest(shap_path)
                old = pd.read_csv(shap_path).set_index("feature").loc[features, "mean_abs_shap"].to_numpy()
                aggregate_error = float(np.max(np.abs(np.abs(grouped).mean(axis=0) - old)))
                require(aggregate_error < 1e-8, f"Published aggregate SHAP mismatch: {aggregate_error}")
            checks.append(dict(stage=stage, split=split, students=len(rows), macro_f1=macro_f1,
                               max_shap_additivity_error=error, max_reference_shap_error=aggregate_error))
            for position, (_, row) in enumerate(rows.iterrows()):
                case_id = f"{split}_{position + 1:04d}_{stage}"
                facts = [dict(feature=f, recorded_value=scalar(row[f]),
                              value_type="categorical_code" if f in categorical else "numeric",
                              dropout_raw_score_contribution=float(grouped[position, j])) for j, f in enumerate(features)]
                cases[split].append(dict(case_id=case_id, feature_specification=TIMING[stage],
                    predicted_class=classes[predictions[position]],
                    model_probabilities=dict(zip(classes, map(float, probabilities[position]))),
                    dropout_raw_score=float(raw[position, 0]), dropout_shap_baseline=float(baseline[0]),
                    predictor_evidence=facts, interpretation_limits=NOTES))
                truth.append(dict(case_id=case_id, StudentID=scalar(row.StudentID), split=split, stage=stage,
                                  actual_outcome=row[target], predicted_class=classes[predictions[position]]))
        fitted_models.append((stage, model.booster_.model_to_string(), prep))
    require(all(digest(p) == before[str(p)] for p in tracked), "A tracked input changed during execution")
    # No files have been written before all numerical checks pass.
    output.mkdir(parents=True, exist_ok=False)
    (output / "llm_inputs").mkdir()
    (output / "evaluation_DO_NOT_PROMPT").mkdir()
    (output / "models").mkdir()
    for split, records in cases.items():
        with (output / "llm_inputs" / f"{split}_evidence.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(truth).to_csv(output / "evaluation_DO_NOT_PROMPT/outcomes_and_id_mapping.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "verification_checks.csv", index=False)
    for stage, model_text, prep in fitted_models:
        (output / "models" / f"{stage}.txt").write_text(model_text, encoding="utf-8")
        dump_json(output / "models" / f"{stage}_preprocessing.json", vars(prep))
    dump_json(output / "manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source), input_sha256=before, counts=counts, seed=123, model_parameters=PARAMS,
        environment={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "lightgbm", "shap"]},
        python=sys.version, class_order=classes, training="Original train split only; no validation/test refitting",
        shap="Tree-path-dependent, raw scores; sum one-hot contributions within each original feature",
        status="Verified evidence export only. No LLM outputs or performance claims.",
        next_gate="Verify category meanings and freeze case-selection/prompt/evaluation protocol on validation data before LLM test runs.",
        limitations=NOTES))
    hashes = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    dump_json(output / "output_sha256.json", hashes)
    print(pd.DataFrame(checks).to_string(index=False), flush=True)
    print(f"SUCCESS: {len(cases['validation'])} validation and {len(cases['test'])} test evidence records.")
    print(f"Saved to {output}. Actual outcomes are ONLY in evaluation_DO_NOT_PROMPT.")


if __name__ == "__main__":
    main()
