# Full-Feature Hyperparameter Tuning Report

Tuning evaluated 150 candidate model configurations across the seven model families. Selection used validation Macro-F1 first, then validation balanced accuracy, then lower validation log-loss.

The tuned outputs are saved separately inside `results/5.Full Feature Final Modeling/Hyp Tuned/` to keep baseline and tuned final-model results easy to compare.

Best tuned validation model: **LightGBM**; validation Macro-F1 = 0.7088, validation balanced accuracy = 0.7957, validation log-loss = 0.5963.