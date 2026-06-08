# Staged Early Warning Model Benchmark Report

This benchmark compares seven model families across three information stages using the same train/validation/test split and the same transformed input representation within each stage. Categorical variables were one-hot encoded and numeric variables were scaled using training-set parameters only.

## Stage Design
- A_pre_enrollment: 24 raw variables, 232 encoded features, 3097/885/442 train/validation/test students.
- B_first_semester: 30 raw variables, 238 encoded features, 3097/885/442 train/validation/test students.
- C_second_semester: 36 raw variables, 244 encoded features, 3097/885/442 train/validation/test students.

## Best Validation Model By Stage
- A_pre_enrollment: Random Forest achieved validation Macro-F1=0.5787, Balanced Accuracy=0.6930, Log-Loss=0.8539.
- B_first_semester: Neural Network / MLP achieved validation Macro-F1=0.6689, Balanced Accuracy=0.7564, Log-Loss=0.6769.
- C_second_semester: Neural Network / MLP achieved validation Macro-F1=0.6995, Balanced Accuracy=0.7807, Log-Loss=0.5829.

## Main Interpretation
Performance improves as academic timing information is added. This supports the early-warning design: pre-enrollment variables provide useful baseline risk information, first-semester academic performance adds substantial predictive value, and second-semester information improves the model further.

## Paper-Relevant Outputs
- Full train/validation/test metrics: `csv/staged_model_comparison.csv`
- Paper-ready metrics table: `csv/paper_ready_train_validation_test_metrics.csv`
- Best validation model table: `csv/best_validation_model_by_stage.csv`
- Train, validation, and test metric pivots: `csv/*_metric_pivot.csv`
- Metric and log-loss plots: `plots/`