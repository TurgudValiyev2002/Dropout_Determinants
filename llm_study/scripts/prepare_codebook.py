"""Build a source-backed codebook; audit training codes only. No model fitting."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

URL = "https://archive.ics.uci.edu/api/dataset?id=697"
PAGE = "https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"


def normalize(name):
    return name.strip().casefold().replace("’", "'")


def parse_mapping(description):
    # Match explicit code-label separators only, not numbers inside labels.
    matches = list(re.finditer(r"(?<!\S)(\d{1,5})\s*[-–]\s+", description))
    if len(matches) < 2:
        return {}
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(description)
        code = match.group(1)
        label = description[match.end():end].strip(" ;\n\r")
        if code in result or not label:
            raise ValueError(f"Ambiguous official category description at code {code}: {description!r}")
        result[code] = label
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot", type=Path, help="Use an existing UCI JSON snapshot; no internet needed")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Choose a NEW output directory outside the source folder")
    base = source / "Dropout_Determinants/results/2.Model Processing"
    meta_path = base / "metadata/feature_sets_metadata.json"
    data_path = base / "csv/modeling_master_dataset_with_splits.csv"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if args.snapshot:
        raw = args.snapshot.read_bytes()
    else:
        print("Reading official UCI metadata ...", flush=True)
        with urlopen(Request(URL, headers={"User-Agent": "Research-codebook-verification/1.0"}), timeout=60) as response:
            raw = response.read()
    payload = json.loads(raw)
    dataset = payload["data"]
    if int(dataset["uci_id"]) != 697:
        raise ValueError("Unexpected UCI dataset")
    documented = {normalize(v["name"]): v for v in dataset["variables"]}
    features = list(dict.fromkeys(f for fs in meta["stage_feature_sets"].values() for f in fs))
    categorical = set(meta["feature_types"]["categorical_features"])
    observed = {f: set() for f in features if f in categorical}
    with data_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Split"] == "train":
                for feature in observed:
                    number = float(row[feature])
                    if not number.is_integer():
                        raise ValueError(f"Noninteger category in {feature}")
                    observed[feature].add(str(int(number)))
    entries, audit = {}, []
    for feature in features:
        record = documented.get(normalize(feature))
        if record is None:
            raise ValueError(f"Feature absent from official metadata: {feature}")
        description = record.get("description") or ""
        entry = {"official_name": record["name"], "official_description": description,
                 "official_units": record.get("units"), "code_labels": {}}
        if feature == "Application order":
            entry["interpretation"] = "Ordinal application preference; retain the recorded rank. Do not relabel code 1 as first choice: the source describes 0 as first choice."
        elif feature in categorical:
            entry["code_labels"] = parse_mapping(description)
            if not entry["code_labels"]:
                raise ValueError(f"No explicit mapping parsed for {feature}; manual review required")
        if feature in observed:
            for code in sorted(observed[feature], key=int):
                label = entry["code_labels"].get(code)
                status = "documented_ordinal" if feature == "Application order" else "verified" if label is not None else "UNRESOLVED"
                audit.append({"feature": feature, "code": code, "status": status,
                              "label": label or ("Keep recorded application rank" if feature == "Application order" else "Meaning not verified; do not infer")})
        entries[feature] = entry
    # Spot checks catch common parsing/inversion mistakes.
    expected = {"Gender": {"0": "female", "1": "male"},
                "Daytime/evening attendance": {"0": "evening", "1": "daytime"},
                "Tuition fees up to date": {"0": "no", "1": "yes"}}
    for feature, codes in expected.items():
        for code, label in codes.items():
            if entries[feature]["code_labels"][code].casefold() != label:
                raise ValueError(f"Failed official mapping spot check: {feature}/{code}")
    if len(entries["Course"]["code_labels"]) != 17:
        raise ValueError("Expected 17 official course codes; check parsing")
    unresolved = [a for a in audit if a["status"] == "UNRESOLVED"]
    codebook = {"dataset_id": 697, "source_api": URL, "source_page": PAGE,
        "dataset_attribution": "Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89",
        "source_license": "CC BY 4.0; https://creativecommons.org/licenses/by/4.0/",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "official_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
        "local_metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        "local_data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "outcome_timing": "At the end of the normal duration of the course; semester specifies the feature window, NOT the outcome deadline.",
        "rules": ["Do not interpret an undocumented code using a different variable's mapping.",
                  "Keep unresolved codes explicit; never guess or silently replace them.",
                  "Preserve original codes and model inputs. This codebook is for explanation only.",
                  "Unknown/blank category labels do not mean the corresponding recorded value is a literal missing cell.",
                  "Financial variable measurement timing remains uncertain; do not claim prospective availability."],
        "audit_scope": "Training split only; no validation or test outcomes used",
        "features": entries, "unresolved_training_codes": unresolved}
    output.mkdir(parents=True, exist_ok=False)
    (output / "uci_697_metadata_snapshot.json").write_bytes(raw)
    (output / "codebook.json").write_text(json.dumps(codebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "training_code_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "code", "status", "label"])
        writer.writeheader()
        writer.writerows(audit)
    print(f"SUCCESS: documented {len(entries)} predictor variables; audited {len(audit)} training feature/code pairs.")
    print(f"Unresolved training feature/code pairs: {len(unresolved)}")
    for item in unresolved:
        print(f"  {item['feature']}: {item['code']} -- do not infer meaning")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
