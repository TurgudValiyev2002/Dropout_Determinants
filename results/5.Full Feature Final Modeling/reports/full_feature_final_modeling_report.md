# Full-Feature Final Modeling Report

This analysis trains one final full-information model family using all available predictors from the prepared modeling dataset. It is not a replacement for the staged early-warning analysis. Instead, it gives a maximum-information benchmark that shows how much predictive performance is available when pre-enrollment, first-semester, and second-semester variables are all used together.

The full-feature matrix uses 36 raw predictors, transformed into 244 encoded/scaled columns using training-set preprocessing only.

## Best Baseline Model

Before tuning, the best validation model was **Neural Network / MLP** with validation Macro-F1 = 0.7001, balanced accuracy = 0.7815, and log-loss = 0.5961.

## Best Tuned Model

After tuning, the best validation model was **LightGBM** with validation Macro-F1 = 0.7088, balanced accuracy = 0.7957, and log-loss = 0.5963. On the held-out test set, this same selected model reached Macro-F1 = 0.7016, balanced accuracy = 0.7892, and log-loss = 0.5902.

## What To Say In The Paper

The staged models should be presented as the early-warning contribution, while the full-feature model should be presented as the final maximum-information benchmark. If the full-feature model only slightly improves over Stage C, that supports the argument that second-semester academic and financial signals already capture most dropout risk. If the full-feature model improves meaningfully, then the additional variables should be discussed as useful for end-of-year or final-risk assessment rather than true early warning.

Recommended tables/figures: full-feature train/validation/test metrics table, before/after tuning comparison table, Stage C vs full-feature best-test comparison, and the top-25 feature-importance plot for the best tuned full-feature model.