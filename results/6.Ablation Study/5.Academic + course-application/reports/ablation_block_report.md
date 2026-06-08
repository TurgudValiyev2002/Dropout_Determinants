# Ablation Block Report: 5.Academic + course-application

This block uses 17 raw predictors transformed into 53 encoded/scaled features.

Best baseline validation model: **Random Forest** with Macro-F1 = 0.6717, balanced accuracy = 0.7660, and log-loss = 0.6543.

Best tuned validation model: **LightGBM** with Macro-F1 = 0.6718, balanced accuracy = 0.7669, and log-loss = 0.6552.
On test, the validation-selected tuned model reached Macro-F1 = 0.6546, balanced accuracy = 0.7539, and log-loss = 0.6417.