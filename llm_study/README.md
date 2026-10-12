# Student outcome explanation experiment

This directory preserves the analysis scripts and saved outputs of an evidence-matched comparison between an LLM and a deterministic template. Both methods receive the same outcome percentages, recorded and decoded feature values, and selected Dropout raw-score contribution directions.

## Where to start

| Location | Contents |
| --- | --- |
| `scripts/` | Frozen experiment runner, evidence exporter, codebook builder, pilot preparation, automatic evaluator, and original Slurm launcher |
| `inputs/source/s1_verified/` | Predictive-analysis input snapshot and original reproduction scripts |
| `outputs/student_evidence_v1/` | Exported validation/test evidence, fitted prediction models, and separate evaluation-only outcome records |
| `outputs/codebook_v1/` | Verified category descriptions and code audit |
| `outputs/final_experiment_v2/` | Frozen protocol, system prompt, 1,326 prompts and matched templates, input hashes, six chunk definitions, and per-case generation records |
| `outputs/final_collection_v2/` | Collected final LLM accounts and collection summary |
| `outputs/automated_evaluation_v1/` | Per-case checks, aggregate tables, paired statistics, analysis manifest, and length/compliance plots |
| Other output directories | Artificial smoke test, validation pilot, input verification, and unused human-review preparation materials |

Root-level copies of the core scripts are retained as supplied; matching copies also exist under `scripts/`. Frozen files are byte-preserved through `.gitattributes`. The original cluster paths in provenance files and the Slurm launcher describe the execution environment; they are not portable defaults. Model weights and runtime caches are not included.

## Completed experiment

There are 442 held-out test students, each represented at three information windows, yielding 1,326 accounts per method. All final LLM accounts were generated and retained. The locally stored NVIDIA Nemotron 3.5 Lightning 30B A3B BF16 model used greedy decoding with thinking disabled, seed 123, and a 512-token output cap. The final prompt followed a validation-only pilot and was frozen before test generation.

All three labelled percentages matched the supplied values in 1,325 LLM accounts and all 1,326 templates. This lexical screen is not a full factual-accuracy score. Mean length was 134.6 words for the LLM and 79.9 for the template; 714 LLM accounts exceeded the requested 120-word limit, compared with none of the templates. Detailed definitions and limitations are in `outputs/automated_evaluation_v1/READ_ME.txt` and `analysis_manifest.json`.

The original protocol includes a planned human review and preselected review cases. That review was not conducted. Review forms and private A/B mappings are preparation artifacts, not completed ratings. Do not treat generation success, numerical matching, or length tests as evidence that the accounts are clearer, more useful, or semantically correct throughout.

## Check frozen inputs without a GPU

Run from the repository root with Python 3.12:

```bash
python -B llm_study/scripts/final_llm_experiment.py run \
  --experiment llm_study/outputs/final_experiment_v2 \
  --model unused-model-path --chunk 0 --check-only
```

Repeat with chunk numbers 1 through 5. `--check-only` verifies the frozen ledger, runner identity, and prompt records before returning; it does not load model weights or write outputs.

## Reproduce the automated evaluation

The evaluator requires NumPy, pandas, and Matplotlib. Recorded versions are listed in the saved analysis manifest. Use a fresh output directory to avoid replacing the archived results:

```bash
python -B llm_study/scripts/evaluate_automated.py \
  --experiment llm_study/outputs/final_experiment_v2 \
  --collection llm_study/outputs/final_collection_v2 \
  --output llm_study/outputs/automated_evaluation_rerun
```

Paired inference uses 442 student-level averages, not 1,326 independent observations. Bootstrap resampling retains each student's three windows and both methods. Proportions in `paired_statistics.csv` are fractions; multiply by 100 to express percentage-point differences.

The public source dataset is UCI Predict Students' Dropout and Academic Success, DOI: https://doi.org/10.24432/C5MC89. Evaluation-only observed outcomes are deliberately separated from LLM inputs. The experiment provides retrospective reporting checks, not evidence of causality, intervention effectiveness, calibrated risk, or distance-education validity.
