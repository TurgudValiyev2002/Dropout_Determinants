# Ablation Block Report: 2.Academic + demographic-financial

This block uses 24 raw predictors transformed into 54 encoded/scaled features.

Best baseline validation model: **Random Forest** with Macro-F1 = 0.6920, balanced accuracy = 0.7796, and log-loss = 0.6324.

Best tuned validation model: **LightGBM** with Macro-F1 = 0.7022, balanced accuracy = 0.7891, and log-loss = 0.6253.
On test, the validation-selected tuned model reached Macro-F1 = 0.6983, balanced accuracy = 0.7851, and log-loss = 0.6213.