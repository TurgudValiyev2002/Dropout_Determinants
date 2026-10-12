"""One artificial-example inference test. No student data; no model updates."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

SYSTEM = """You explain student-outcome model predictions to university advisers.
Use only the supplied evidence. Write clear, respectful English.
Rules:
- Explain the prediction, not the student's actual outcome.
- Model probabilities are estimates, not certainties.
- SHAP contributions explain the model's raw score. They are not probability changes, causal effects, or evidence that an intervention works.
- Positive SHAP contributions increase the Dropout raw score relative to its baseline; negative contributions decrease it.
- Do not invent personal circumstances or explanations for recorded values.
- Do not describe the student as a failure or assume they will leave.
- Do not recommend actions unless explicitly requested.
- If evidence is insufficient, acknowledge the limitation."""

USER = """This is an artificial example for testing, not a real student.
Prediction time: after the first semester.
Model probabilities:
- Dropout: 65%
- Enrolled: 15%
- Graduate: 20%
Recorded academic information:
- First-semester curricular units enrolled: 6
- First-semester curricular units approved: 2
- First-semester average grade: 9 out of 20
Dropout-class SHAP contributions, in raw-score units:
- Curricular units approved: +0.80
- Average grade: +0.50
- Curricular units enrolled: -0.10
SHAP baseline raw score: -0.40
Final Dropout raw score: +0.80
These three contributions are the complete explanation for this artificial example.
Write an explanation of no more than 120 words for a university adviser.
Include:
1. The predicted outcome and its uncertainty.
2. The main factors raising the Dropout raw score, and the factor lowering it.
3. A brief statement that this explanation does not establish causes.
Do not convert SHAP contributions into percentage-point changes.
Do not infer why the student received these grades."""


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model_path, output = args.model.resolve(), args.output.resolve()
    if not model_path.is_dir():
        raise ValueError("Local model directory not found")
    if output == model_path or model_path in output.parents:
        raise ValueError("Output cannot be inside the model directory")
    # Refuse overwrites, including a previous failed test.
    output.mkdir(parents=True, exist_ok=False)
    sys.dont_write_bytecode = True
    for variable, suffix in {
        "HF_HOME": "hf", "XDG_CACHE_HOME": "xdg", "TORCH_HOME": "torch",
        "TORCH_EXTENSIONS_DIR": "torch_extensions", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "NUMBA_CACHE_DIR": "numba",
    }.items():
        location = output / "runtime_cache" / suffix
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    save(output / "messages.json", messages)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.device_count() != 2:
            raise RuntimeError("This test expects exactly two allocated GPUs")
        torch.manual_seed(123)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt")
        if encoded["input_ids"].shape[-1] > 2048:
            raise RuntimeError("Unexpectedly long artificial prompt")
        (output / "rendered_prompt.txt").write_text(
            tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False), encoding="utf-8")
        print("Loading BF16 weights across two GPUs (no CPU/disk offload) ...", flush=True)
        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, dtype=torch.bfloat16, device_map="balanced",
            max_memory={0: "38GiB", 1: "38GiB", "cpu": 0}, attn_implementation="eager")
        model.eval()
        placement = {name: str(device) for name, device in model.hf_device_map.items()}
        save(output / "device_map.json", placement)
        if any(device in {"cpu", "disk", "meta"} for device in placement.values()):
            raise RuntimeError("Unexpected offload: this test requires GPU-only model placement")
        load_seconds = time.perf_counter() - started
        device = model.get_input_embeddings().weight.device
        encoded = encoded.to(device)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        print("Model loaded. Generating one artificial explanation ...", flush=True)
        for i in range(2):
            torch.cuda.synchronize(i)
            torch.cuda.reset_peak_memory_stats(i)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**encoded, max_new_tokens=256, do_sample=False,
                use_cache=True, pad_token_id=pad_id)
        for i in range(2):
            torch.cuda.synchronize(i)
        generation_seconds = time.perf_counter() - started
        new_ids = generated[0, encoded["input_ids"].shape[-1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True)
        (output / "answer.txt").write_text(answer + "\n", encoding="utf-8")
        config_path = model_path / "config.json"
        save(output / "run_metadata.json", {
            "model_path": str(model_path), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate"]},
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "python": sys.version,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(2)],
            "settings": {"dtype": "bfloat16", "device_map": "balanced", "attention": "eager",
                "enable_thinking": False, "do_sample": False, "max_new_tokens": 256, "seed": 123},
            "prompt_tokens": int(encoded["input_ids"].shape[-1]), "generated_tokens": len(new_ids),
            "token_limit_reached": len(new_ids) >= 256, "answer_word_count": len(answer.split()),
            "load_seconds": load_seconds, "generation_seconds": generation_seconds,
            "peak_allocated_gib": [torch.cuda.max_memory_allocated(i) / 1024**3 for i in range(2)],
            "scope": "Artificial smoke test only, not a scientific evaluation or throughput benchmark",
        })
        if not answer.strip():
            raise RuntimeError("Generation returned an empty answer")
        print("\n--- MODEL ANSWER ---\n" + answer, flush=True)
        print(f"\nSUCCESS: saved artificial test to {output}", flush=True)
        if len(new_ids) >= 256:
            print("WARNING: token limit reached; answer may be incomplete.", flush=True)
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
