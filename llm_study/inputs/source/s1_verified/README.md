# Supplementary Analysis S1

This archive accompanies the 9 October 2026 manuscript revision. It is a local supplement prepared for author review and submission, not a claim of public deposition.

## Contents

- `Dropout_Determinants/results/2.Model Processing`: analysis input with fixed row-level train/validation/test membership and explicit stage, domain, and variable-type metadata.
- `Dropout_Determinants/results/4.Hyperparameter Tuning/csv`: historical candidate-validation scores and selected settings. The original tuning search is documented, not rerun by this archive.
- `Dropout_Determinants/models`: exact preprocessing, model factories, and metric definitions used by the recorded-settings refits.
- `critical_analysis.py`, `consistency_analysis_07_10.py`: refitted primary estimates, matched-training baselines, fixed-settings repeated cross-validation and domain comparisons, and Figures 3–6.
- `submission_checks_09_10.py`: financial-status exclusion and expanded k=2,...,8 clustering analysis, including prediction vectors, feature lists, cluster assignments and numerical summaries.
- `consistency_07_10`, `submission_checks_09_10`: results corresponding to the manuscript; CSVs retain full precision.
- `requirements.txt`: pinned packages from the executed analysis environment. Python 3.12.14 was used; see `consistency_07_10/environment.json`.
- `SHA256SUMS.json`: checksums of every included file except the checksum manifest itself.

## Rerun

Extract the archive to an ordinary writable directory. Use a clean Python 3.12 environment:

```text
python -m venv .venv
# Activate the environment according to your operating system.
python -m pip install -r requirements.txt
python run_all.py
```

The full run can take several minutes or longer depending on hardware. It trains the recorded model configurations and regenerates the primary and supplemental outputs in place. Keep a copy of the archive for comparison. It does not load model pickles, connect to private services, download new data, or use absolute paths from the author's computer.

For a quick input and model-fitting check, run `python verify_inputs.py`. This independently refits the three primary LightGBM models and checks their Macro-F1 against the saved results. This smoke test is not a full reproduction of bootstrap intervals, repeated cross-validation, SHAP, or clustering.

## Interpretation and conventions

The fixed split contains 3,097 training, 885 validation, and 442 test students. StudentID is a local record identifier and is never a predictor. The three outcome classes are ordered Dropout=0, Enrolled=1, Graduate=2. There are 36 original predictors plus Target; one-hot encoding expands matrix width. Balanced accuracy is macro recall.

The original validation-selected LightGBM settings are retained. Refitted validation scores and test scores are not used to select a replacement model. Cross-validation fixes these previously chosen settings and therefore is resampling sensitivity, not nested evaluation of the entire selection procedure. Test bootstrap intervals are conditional on fitted models and the sampled students. Repeated reuse of this split must not be described as independent confirmation.

The exclusion sensitivity removes Tuition fees up to date, Debtor, and Scholarship holder at all three stages, retaining 21, 27, and 33 original predictors. Hyperparameters are not retuned. Canonical recorded-model intervals are reused from the primary output, so the paper does not report different Monte Carlo intervals for the same model in different tables. New excluded-model and paired-exclusion intervals use seed 123, 3,000 marginal and 5,000 paired resamples. Unknown timestamps and dropout dates remain unresolved; exclusion is not proof that all remaining features were prospectively available.

Descriptive clustering uses the full sample, excluding Target and predicted risk. The maximum silhouette among k=2,...,8 selects three profiles, whereas Calinski-Harabasz favours two. This disagreement is reported. Optimization stability across seeds is not resampling stability or external validation. Numerical profile labels are descriptive, not intervention categories.

SHAP explains the Dropout-class raw score. For each student, signed one-hot contributions belonging to a variable are summed before averaging absolute values. Figures 5 and 6(c) use the same fitted model. Numeric point colours in Figure 6 denote within-variable ranks, not a shared physical scale.

## Data provenance and attribution

Source: Realinho, V., Vieira Martins, M., Machado, J., and Baptista, L. Predict Students' Dropout and Academic Success. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

The input included here is a derivative of the public source with reproducibility identifiers, outcome encodings, and fixed split membership. Consult the source repository for its licence and attribution terms. No private institutional records have been added. Original project: https://github.com/TurgudValiyev2002/Dropout_Determinants

Closest phased predecessor: Martins, M. V., Baptista, L., Machado, J., and Realinho, V. (2023). Multi-Class Phased Prediction of Academic Performance and Dropout in Higher Education. Applied Sciences, 13(8), 4702. https://doi.org/10.3390/app13084702 . Its population and evaluation differ; its scores should not be treated as a directly matched benchmark.

## Limits of this package

This package reproduces the revised recorded-settings analyses, not the historical environment or full historical tuning search. Floating-point and hardware-dependent differences may occur. Figures 1–2 are preserved descriptive illustrations from the existing manuscript and are not regenerated here. No temporal split, timestamp recovery, prospective intervention trial, or institutional ethics determination is created by these scripts.
