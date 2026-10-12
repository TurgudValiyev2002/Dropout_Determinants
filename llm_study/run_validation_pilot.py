"""Generate 20 validation explanations; preserve every case and failure.

Reuse the tested BF16/two-GPU loading method. Read local weights only.
--dry-run checks pilot inputs without importing torch or creating output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def check(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def no_forbidden_keys(value):
    forbidden = {"actual_outcome", "StudentID", "Target", "DropoutBinary", "DropoutLabel", "y"}
    if isinstance(value, dict):
        check(not (set(value) & forbidden), "Outcome/identifier key found in prompts")
        for item in value.values():
            no_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            no_forbidden_keys(item)


def load_pilot(root):
    protocol_path = root / "protocol.json"
    prompt_path = root / "pilot_prompts.jsonl"
    ledger = json.loads((root / "output_sha256.json").read_text(encoding="utf-8"))
    for path in (protocol_path, prompt_path):
        check(sha(path) == ledger[path.name], f"Hash mismatch: {path.name}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check(protocol["stage"] == "B_first_semester" and protocol["n"] == 20, "Unexpected pilot protocol")
    check(len(records) == 20 and len({r['case_id'] for r in records}) == 20, "Expected 20 unique cases")
    check(Counter(r['evidence']['predicted_class'] for r in records) == {"Dropout": 8, "Enrolled": 6, "Graduate": 6}, "Wrong case quotas")
    check(protocol["planned_inference"] == {"enable_thinking": False, "do_sample": False, "max_new_tokens": 512, "seed": 123}, "Unexpected inference settings")
    for record in records:
        check(record["case_id"].startswith("validation_") and record["case_id"].endswith("_B_first_semester"), "Nonpilot case found")
        check(record["case_id"] == record["evidence"]["case_id"], "Case ID mismatch")
        no_forbidden_keys(record)
        messages = record["messages"]
        check(len(messages) == 2 and [m['role'] for m in messages] == ['system', 'user'], "Unexpected message roles")
        check("no more than 120 words" in messages[0]["content"], "Wrong word limit")
        # The actual messages must contain exactly the evidence we audited.
        supplied = json.loads(messages[1]["content"].split("\n", 1)[1])
        check(supplied == record["evidence"], "Message/evidence mismatch")
    return protocol, records, {p.name: sha(p) for p in (protocol_path, prompt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pilot, model_path, output = args.pilot.resolve(), args.model.resolve(), args.output.resolve()
    check(not output.exists(), "Output exists; choose a NEW versioned directory. Nothing was overwritten.")
    check(not output.is_relative_to(pilot) and not output.is_relative_to(model_path), "Output must be outside pilot/model directories")
    protocol, records, input_hashes = load_pilot(pilot)
    if args.dry_run:
        print("PASS: 20 validation-only prompts, quotas, hashes, evidence consistency, word limit, and inference settings.")
        print("Dry run only: no model loaded and no output written.")
        return
    check(model_path.is_dir(), "Local model directory not found")
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    for variable, suffix in {
        "HF_HOME": "hf", "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "NUMBA_CACHE_DIR": "numba",
    }.items():
        cache = output / "runtime_cache" / suffix
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    (output / "cases").mkdir()
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
        "scope": "Validation development pilot only; not final evaluation or predictive improvement evidence",
        "input_sha256": input_hashes, "script_sha256": sha(Path(__file__)),
        "model_path": str(model_path), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "settings": {**protocol["planned_inference"], "dtype": "bfloat16", "device_map": "balanced", "attention": "eager"}}
    save(output / "run_metadata.json", metadata)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        check(torch.cuda.device_count() == 2, "Exactly two allocated GPUs are required")
        torch.manual_seed(123)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        encoded_cases = []
        for record in records:
            encoded = tokenizer.apply_chat_template(record["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_dict=True, return_tensors="pt")
            check(encoded["input_ids"].shape[-1] <= 4096, "Unexpected prompt length; no silent truncation allowed")
            encoded_cases.append(encoded)
            save(output / "cases" / f"{record['case_id']}_input.json", record)
            (output / "cases" / f"{record['case_id']}_rendered_prompt.txt").write_text(
                tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False), encoding="utf-8")
        print("Loading BF16 model once across two GPUs ...", flush=True)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map="balanced", max_memory={0: "38GiB", 1: "38GiB", "cpu": 0}, attn_implementation="eager")
        model.eval()
        placement = {k: str(v) for k, v in model.hf_device_map.items()}
        check(not any(v in {"cpu", "disk", "meta"} for v in placement.values()), "Unexpected model offload")
        save(output / "device_map.json", placement)
        metadata.update(load_seconds=time.perf_counter() - started, python=sys.version,
            environment={p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate"]},
            gpu_names=[torch.cuda.get_device_name(i) for i in range(2)],
            model_file_sha256={p.name: sha(p) for p in [model_path / "config.json", model_path / "tokenizer_config.json", model_path / "model.safetensors.index.json"] if p.is_file()})
        save(output / "run_metadata.json", metadata)
        input_device = model.get_input_embeddings().weight.device
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        results, abort = [], False
        with (output / "llm_explanations.jsonl").open("w", encoding="utf-8") as handle:
            for position, (record, encoded) in enumerate(zip(records, encoded_cases), 1):
                result = {"case_id": record["case_id"], "method": "nemotron_bf16_greedy_v1", "status": "not_attempted",
                          "text": None, "word_count": None, "over_word_limit": None, "token_limit_reached": None}
                if not abort:
                    print(f"Generating {position}/20: {record['case_id']}", flush=True)
                    started = time.perf_counter()
                    try:
                        encoded = encoded.to(input_device)
                        for i in range(2):
                            torch.cuda.synchronize(i)
                            torch.cuda.reset_peak_memory_stats(i)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            generated = model.generate(**encoded, max_new_tokens=512, do_sample=False, use_cache=True, pad_token_id=pad_id)
                        for i in range(2):
                            torch.cuda.synchronize(i)
                        elapsed = time.perf_counter() - started
                        ids = generated[0, encoded["input_ids"].shape[-1]:].tolist()
                        answer = tokenizer.decode(ids, skip_special_tokens=True)
                        words = len(answer.split())
                        result.update(status="generated" if answer.strip() else "empty", text=answer, word_count=words,
                            over_word_limit=words > 120, token_limit_reached=len(ids) >= 512,
                            generation_seconds=elapsed, prompt_tokens=int(encoded["input_ids"].shape[-1]),
                            generated_tokens=len(ids), generated_token_ids=ids,
                            peak_allocated_gib=[torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(2)])
                        (output / "cases" / f"{record['case_id']}_answer.txt").write_text(answer + "\n", encoding="utf-8")
                    except Exception:
                        result.update(status="failed", error=traceback.format_exc(), elapsed_to_error_seconds=time.perf_counter() - started)
                        abort = True  # Preserve the error; do not silently retry or continue a damaged CUDA context.
                else:
                    result["reason"] = "Not attempted after an earlier generation failure"
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                results.append(result)
        summary = {"records": len(results), "statuses": dict(Counter(r["status"] for r in results)),
            "over_word_limit": sum(r["over_word_limit"] is True for r in results),
            "token_limit_reached": sum(r["token_limit_reached"] is True for r in results),
            "quality_status": "Not yet assessed; generated does NOT mean factually correct or scientifically validated"}
        save(output / "summary.json", summary)
        metadata.update(status="completed_with_generation_failure" if abort else "completed", finished_utc=datetime.now(timezone.utc).isoformat())
        save(output / "run_metadata.json", metadata)
        save(output / "result_sha256.json", {p.name: sha(p) for p in [output / "llm_explanations.jsonl", output / "summary.json", output / "run_metadata.json"]})
        print("\nPILOT FINISHED\n" + json.dumps(summary, indent=2), flush=True)
        print(f"Saved to {output}", flush=True)
        if abort:
            print("Generation failed; all 20 case statuses were preserved. Inspect the saved case error before another run.", flush=True)
            raise SystemExit(1)
    except Exception:
        metadata.update(status="error", error=traceback.format_exc())
        save(output / "run_metadata.json", metadata)
        (output / "error.txt").write_text(metadata["error"], encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
