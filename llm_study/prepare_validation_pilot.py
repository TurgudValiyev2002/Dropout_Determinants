"""Prepare 20 validation-only prompts and matched template explanations.

No inference, retraining, actual-outcome access, or test-file access.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

SEED = 123
STAGE = "B_first_semester"
QUOTAS = {"Dropout": 8, "Enrolled": 6, "Graduate": 6}
SYSTEM = """You explain student-outcome model predictions to university advisers.
Use only the supplied evidence and write clear, respectful English.
Explain the model prediction, not the student's actual outcome.
The model uses first-semester features to predict the outcome at the end of the normal course duration. Do not say graduation or dropout occurs at the end of the first semester.
Probabilities are uncalibrated estimates, not certainties.
All SHAP contributions provided here explain only the Dropout class raw score. They do not directly explain the other class scores, change probabilities by that amount, establish causes, or demonstrate intervention benefits.
A positive contribution raises the Dropout raw score relative to its baseline; a negative contribution lowers it.
The supplied contributions are a selected subset, not the complete explanation. Unshown contributions are summarized separately.
Treat each contribution as belonging to the supplied recorded feature value. Do not claim what changing that value would do, or infer a general trend from one observation.
Do not invent motives, finances, family circumstances, meanings of undocumented codes, or additional facts. Do not infer why a student obtained a grade.
Unknown or blank code labels mean the category meaning is unknown or blank, not that its raw value was absent.
Financial measurement timing is uncertain; do not claim these records establish a prospectively available early-warning system.
Do not recommend actions, stereotype students, or describe them as failures.
Write two or three short paragraphs, no more than 120 words. State the predicted category and all three probabilities, describe every selected contributor with its recorded value and direction, and acknowledge uncertainty and non-causality. Avoid raw-score arithmetic in the prose; contributions are available as evidence, not percentage-point effects.
Include this brief limitation: financial-record timing is uncertain.
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def order_key(record):
    return hashlib.sha256(f"{SEED}:{record['case_id']}".encode()).hexdigest()


def decode(item, codebook):
    feature = item["feature"]
    raw = item["recorded_value"]
    entry = codebook["features"][feature]
    value = str(raw)
    status = "numeric"
    if raw is None:
        value, status = "not recorded", "missing"
    elif item["value_type"] == "categorical_code":
        check(float(raw).is_integer(), f"Noninteger category: {feature}")
        code = str(int(raw))
        if feature == "Application order":
            value, status = f"recorded application rank {code}", "documented_ordinal"
        elif code in entry["code_labels"]:
            value, status = entry["code_labels"][code], "verified_code"
        else:
            value, status = f"code {code}, meaning not verified", "unresolved_code"
    elif "grade" in feature.lower():
        # Preserve raw zeros; do not interpret a zero as a missing grade.
        scale = 20 if "sem" in feature else 200
        value = f"{raw:g} on the documented 0–{scale} scale"
    elif isinstance(raw, (int, float)):
        value = f"{raw:g}"
    return {"feature": feature, "recorded_value": raw, "decoded_value": value,
            "decoding_status": status,
            "dropout_raw_score_contribution": item["dropout_raw_score_contribution"],
            "direction": "raises" if item["dropout_raw_score_contribution"] > 0 else "lowers"}


def template(evidence):
    p = evidence["model_probabilities_percent"]
    text = (f"Using the first-semester feature specification, the model predicts {evidence['predicted_class']}: "
            f"Dropout {p['Dropout']:.1f}%, Enrolled {p['Enrolled']:.1f}%, and Graduate {p['Graduate']:.1f}%. "
            "These are uncalibrated estimates of the outcome at the end of the normal course duration, not certainties. ")
    for item in evidence["selected_contributors"]:
        text += f"{item['feature']} (recorded: {item['decoded_value']}) {item['direction']} the Dropout model score relative to its baseline. "
    return text + ("These are selected contributors, not the complete explanation or evidence of causes. "
                   "Financial-record timing is uncertain.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, codebook_path, output = args.evidence_root.resolve(), args.codebook.resolve(), args.output.resolve()
    check(not output.exists(), "Output exists; choose a new versioned directory")
    check(not output.is_relative_to(root) and not output.is_relative_to(codebook_path.parent), "Keep output outside the evidence/codebook directories")
    evidence_path = root / "llm_inputs/validation_evidence.jsonl"
    manifest_path = root / "manifest.json"
    ledger_path = root / "output_sha256.json"
    ledger = {k.replace("\\", "/"): v for k, v in json.loads(ledger_path.read_text(encoding="utf-8")).items()}
    check(sha(evidence_path) == ledger["llm_inputs/validation_evidence.jsonl"], "Validation evidence hash mismatch")
    check(sha(manifest_path) == ledger["manifest.json"], "Evidence manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    codebook = json.loads(codebook_path.read_text(encoding="utf-8"))
    hashes = set(manifest["input_sha256"].values())
    check(codebook["local_data_sha256"] in hashes and codebook["local_metadata_sha256"] in hashes, "Codebook and model evidence use different sources")
    records = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(len(records) == 2655 and len({r['case_id'] for r in records}) == len(records), "Unexpected validation record count or duplicate IDs")
    forbidden = {"actual_outcome", "StudentID", "Target", "DropoutBinary", "DropoutLabel", "y"}
    for record in records:
        check(record["case_id"].startswith("validation_"), "Nonvalidation record found")
        check(not (set(record) & forbidden), "Outcome/identifier leakage")
        check(not ({f['feature'] for f in record['predictor_evidence']} & forbidden), "Unsafe predictors")
    pool = [r for r in records if r["case_id"].endswith("_" + STAGE)]
    check(len(pool) == 885, "Expected 885 first-semester validation cases")
    selected = []
    for label, quota in QUOTAS.items():
        group = sorted((r for r in pool if r["predicted_class"] == label), key=order_key)
        check(len(group) >= quota, f"Insufficient cases predicted as {label}")
        selected.extend(group[:quota])
    selected.sort(key=order_key)
    prompts, baselines, review_rows, audit = [], [], [], []
    for record in selected:
        facts = record["predictor_evidence"]
        check(len({f['feature'] for f in facts}) == len(facts), "Duplicate predictor names")
        check(all(math.isfinite(f['dropout_raw_score_contribution']) for f in facts), "Nonfinite contribution")
        total = sum(f["dropout_raw_score_contribution"] for f in facts)
        check(abs(total + record["dropout_shap_baseline"] - record["dropout_raw_score"]) < 1e-6, "SHAP reconstruction failed")
        # Equal evidence budget in both arms: two largest positive and one
        # largest-magnitude negative contributor, ties broken by feature name.
        positive = sorted((f for f in facts if f["dropout_raw_score_contribution"] > 0), key=lambda f: (-f["dropout_raw_score_contribution"], f["feature"]))[:2]
        negative = sorted((f for f in facts if f["dropout_raw_score_contribution"] < 0), key=lambda f: (f["dropout_raw_score_contribution"], f["feature"]))[:1]
        chosen = positive + negative
        decoded = [decode(f, codebook) for f in chosen]
        probabilities = record["model_probabilities"]
        check(set(probabilities) == {"Dropout", "Enrolled", "Graduate"}, "Unexpected probability labels")
        check(all(0 <= p <= 1 for p in probabilities.values()) and abs(sum(probabilities.values()) - 1) < 1e-8, "Invalid probabilities")
        evidence = {"case_id": record["case_id"], "feature_window": "First-semester feature specification; financial measurement timing uncertain",
                    "outcome_time": codebook["outcome_timing"], "predicted_class": record["predicted_class"],
                    "model_probabilities_percent": {k: round(100 * probabilities[k], 1) for k in QUOTAS},
                    "selected_contributors": decoded, "dropout_shap_baseline": record["dropout_shap_baseline"],
                    "dropout_raw_score": record["dropout_raw_score"],
                    "unshown_contributions_sum": total - sum(f["dropout_raw_score_contribution"] for f in chosen),
                    "shap_units": "Dropout raw-score units, NOT percentage points"}
        # Rounding can make displayed percentages sum to 99.9 or 100.1.
        evidence["probability_note"] = "Percentages rounded independently to one decimal place; totals may differ slightly from 100%."
        baseline = template(evidence)
        check(len(baseline.split()) <= 120, "Template exceeds the shared 120-word limit")
        prompts.append({"case_id": record["case_id"], "evidence": evidence,
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": "Explain the following recorded model evidence.\n" + json.dumps(evidence, ensure_ascii=False)}]})
        baselines.append({"case_id": record["case_id"], "method": "fixed_template_v1", "text": baseline, "word_count": len(baseline.split())})
        for item in decoded:
            audit.append({"case_id": record["case_id"], **item})
        review_rows.append({"case_id": record["case_id"], "predicted_class": record["predicted_class"],
                            "template_words": len(baseline.split()), "unresolved_selected_codes": sum(f["decoding_status"] == "unresolved_code" for f in decoded)})
    protocol = {"purpose": "Development pilot, NOT final scientific evaluation", "created_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Can an LLM explain student-risk predictions more clearly than a fixed template while remaining faithful to the same evidence?",
        "stage": STAGE, "seed": SEED, "n": 20, "quotas_by_PREDICTED_class": QUOTAS,
        "selection": "Within predicted class, sort SHA256 of '123:case_id'; take quota. Never use actual outcomes.",
        "generalization_warning": "Deliberately balanced pilot; not representative of population prevalence and not evidence of predictive improvement.",
        "shared_evidence": "Two largest positive and one largest-magnitude negative Dropout contributors, when available; identical evidence in both arms. Other contributions retained as a scalar remainder.",
        "length": "Both arms limited to 120 words; record realized length and inspect imbalance before final evaluation.",
        "planned_inference": {"enable_thinking": False, "do_sample": False, "max_new_tokens": 512, "seed": SEED},
        "checks_before_final_run": ["Inspect numerical accuracy, contributor coverage/direction, category decoding, and invented claims.",
            "Check outcome-time errors, causal/intervention claims, sensitive-attribute stereotyping, and financial-timing caveat.",
            "Review clarity and usefulness; record failed, empty, overlength, or truncated outputs without silently dropping them.",
            "Pilot ratings are developmental. Freeze the revised protocol before any test-case inference.",
            "Final human review should hide method labels and randomize order; a separate script will prepare it.",
            "Do not treat a successful pilot as evidence of improved retention, causality, or distance-education validity."],
        "inputs": {"validation_sha256": sha(evidence_path), "manifest_sha256": sha(manifest_path), "codebook_sha256": sha(codebook_path), "script_sha256": sha(Path(__file__))}}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "protocol.json", protocol)
    (output / "system_prompt.txt").write_text(SYSTEM, encoding="utf-8")
    for name, items in [("pilot_prompts.jsonl", prompts), ("template_explanations.jsonl", baselines)]:
        with (output / name).open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
    for name, items in [("selected_cases.csv", review_rows), ("selected_evidence_audit.csv", audit)]:
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    save(output / "output_sha256.json", {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    print("SUCCESS: prepared 20 validation-only cases, 20 prompts, and 20 fixed-template explanations.")
    print("Predicted-class quotas: Dropout=8, Enrolled=6, Graduate=6")
    print(f"Unresolved selected category codes: {sum(r['unresolved_selected_codes'] for r in review_rows)}")
    print("No LLM inference, test-file access, or actual-outcome access performed.")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
