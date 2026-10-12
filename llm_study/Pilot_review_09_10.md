# Validation pilot review — 9 October 2026

## Overall assessment: Needs revision before final evaluation

All 20 uploaded LLM explanations, corresponding prompt evidence, and fixed templates were reviewed. Inputs have matching unique case IDs. This is a developmental audit by the assistant, not a blinded human evaluation, inter-rater study, or evidence of improved student retention.

## Verified findings

- All 20 generated answers reproduce the three supplied probability/class assignments (manual attribution review, supported by numerical checks).
- All 20 include a financial-record timing caveat.
- No reversal of a supplied contributor's direction was found among the 59 mentioned contributors. One of the 60 supplied contributors was omitted entirely.
- One answer exceeds 120 words: validation_0439, 127 words. No answer has the runner's token-limit flag.
- Mean length is 94.2 words for LLM answers and 94.6 for templates. Similar average length does not establish similar readability or usefulness.
- Eight answers omit an explicit non-causality statement, and four omit the word/qualification 'uncalibrated'. These omissions were read manually, not inferred solely from keyword matching.
- None of the LLM answers explicitly preserves the normal-course-duration outcome horizon. They do not repeat the artificial test's specific 'graduation after the first semester' error, but the intended horizon should be stated.
- The fixed templates include all three selected contributors and their decoded recorded values for all 20 cases. They remain repetitive and need independent clarity assessment; completeness alone does not make them more useful.

## Material issues

1. **Raw-score interpretation:** validation_0264 uses the lower Dropout raw score to justify the Graduate prediction. A Dropout-only score does not distinguish Graduate from Enrolled. validation_0320 interprets a lower Dropout raw score as reduced likelihood relative to a baseline without a baseline probability. In a multiclass model, a class probability also depends on the other class scores. Do not equate the sign or baseline difference of one raw score with the probability or identity of the winning class.
2. **Incomplete evidence:** validation_0133 omits credited curricular units = 4, whose positive contribution is about +0.378. validation_0351 mentions the GDP contribution but omits its recorded value 0.79. validation_0643 mentions credited units but omits the recorded count 6.
3. **Guardrails are inconsistent:** eight missing non-causality statements, four missing calibration qualifications, and one overlength response. Several answers use 'will Graduate/remain Enrolled', despite later uncertainty caveats. Prefer 'The highest model probability is for ...'. Statements that probabilities 'do not predict the actual outcome' should instead say that predictions do not establish or guarantee the actual outcome.
4. **Unsupported qualitative interpretation:** validation_0305 calls a previous qualification grade of 121/200 'high', although no reference threshold is supplied. Some answers say 'only five' approved units or 'current' unemployment, adding evaluations or temporal interpretations not established by the prompt. Preserve neutral recorded values.

## Case-by-case manual notes

Case suffixes below all refer to validation_*_B_first_semester. Probabilities are correctly assigned in every case. Notes are developmental flags, not independent statistical labels.

| Case | Main notes |
| --- | --- |
| 0264 | All three contributors mentioned; invalid inference from Dropout raw score to Graduate; unnecessary raw-score arithmetic. |
| 0351 | GDP recorded value omitted; arithmetic crowds out explanation; contradictory wording about not predicting the actual outcome. |
| 0348 | Contributors/values and directions retained; no non-causality or uncalibrated qualification; 'only five' is unnecessary judgment. |
| 0201 | Contributors retained; attributing the overall reduction strongly to academic progress overreaches the partial explanation because the unshown net reduction is larger. |
| 0133 | Credited curricular units = 4 omitted entirely; two factors described as if they were the selected set. |
| 0028 | Contributors retained; non-causality and uncalibrated qualifications missing. |
| 0363 | Contributors retained; 'tuition not up to date' is more precise than 'unpaid tuition'; avoid saying predictions do not predict outcomes. |
| 0529 | Contributors retained; non-causality present; avoid categorical 'will remain' wording. |
| 0645 | Contributors retained; non-causality present; neutral 'Displaced: yes' is safer than expanding its meaning beyond documentation. |
| 0320 | Contributors retained; raw-score-to-baseline probability inference unsupported; non-causality statement absent. |
| 0305 | Contributors retained; unsupported 'high' grade and 'current' unemployment descriptions; non-causality statement absent. |
| 0491 | Contributors retained; non-causality statement absent; incorrect suggestion that model estimates are not predictive at all. |
| 0218 | Contributors retained; non-causality and uncalibrated qualifications missing; shortest answer, 58 words. |
| 0439 | Contributors retained; 127 words; unnecessary score arithmetic and 'supports' interpretation. |
| 0643 | Credited units' recorded count 6 omitted; non-causality statement absent. |
| 0114 | Contributors retained; uncertainty and non-causality present; raw-score arithmetic unnecessary. |
| 0697 | Contributors retained; use age **at enrollment**, and 'tuition not up to date'; avoid 'only five'. |
| 0672 | Contributors retained; non-causality present; uncalibrated qualification omitted; avoid suggesting a three-factor subset is the full driver of the prediction. |
| 0055 | Contributors retained; uncertainty and non-causality present; no major direction/recorded-value error identified. |
| 0375 | Contributors retained; non-causality statement absent; raw-score arithmetic unnecessary. |

## Recommended next experiment

Revise the prompt and presentation layer, preserving v1 inputs and outputs. Use the same 20 validation cases for a paired development recheck; do not call this an independent evaluation. Do not inspect or generate test explanations yet.

- Keep the same case-selection rule and selected contributors; do not remove difficult or sensitive-feature cases just to improve scores.
- Give both methods identical selected evidence. For the next presentation version, keep full raw-score diagnostics in audit files but remove baseline, final raw score, and scalar remainder from the writer's prompt. These invite invalid arithmetic and probability interpretations. State clearly that the supplied contributors are a subset.
- Require every selected feature and recorded value, neutral wording, and an explicit prediction horizon.
- Use a mandatory short closing statement covering uncalibrated estimates, non-causality, and uncertain financial timing. Disallow raw-score-to-probability reasoning and claims about which other class a Dropout contribution explains.
- Retain the 120-word limit. Record rather than truncate or delete any overlength answer. If a correction/retry policy is later introduced, define and log it in advance and distinguish first-pass compliance from corrected output.
- Preserve the fixed template as a real comparison, with equal evidence and required caveats. Apply any shared presentation changes to both methods and version them.
- After the revised validation audit passes, freeze prompts, evidence selection, model settings, failure handling, and the final sampling/rating protocol. Obtain blinded, randomized human ratings for clarity and usefulness; this assistant audit is not a substitute.

## Reproducibility and limitations

The companion script `audit_uploaded_pilot.py` recomputes file counts, case alignment, prompt/evidence consistency, percentage-number presence, lengths, token flags, contributor arithmetic, template coverage, and descriptive latency. It does not automatically establish semantic correctness. Manual notes above supply the semantic review.

Uploaded source SHA256 values:

- LLM answers: `9de3a491adbfa5b70dd5700a392859cd0162cbf50f03ab901b06e7a69e257af6`
- Prompts: `752012b0ef405db9d6c6de20b6e8f74988405480bb94650f6c1e12fc2d13d447`
- Templates: `d48519f85d2811ab6bdbb1e1a97bcf46e23776fc888831e6384329f7574d1298`

These hashes identify the uploaded pasted files, not an independently checked cluster-run manifest. Generation timing is descriptive: median 5.30 seconds, with first case 30.54 seconds. This is not a controlled throughput benchmark.
