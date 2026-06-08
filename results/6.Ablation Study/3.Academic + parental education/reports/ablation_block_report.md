# Ablation Block Report: 3.Academic + parental education

This block uses 18 raw predictors transformed into 144 encoded/scaled features.

Best baseline validation model: **Random Forest** with Macro-F1 = 0.6522, balanced accuracy = 0.7517, and log-loss = 0.6835.

Best tuned validation model: **LightGBM** with Macro-F1 = 0.6653, balanced accuracy = 0.7627, and log-loss = 0.6581.
On test, the validation-selected tuned model reached Macro-F1 = 0.6695, balanced accuracy = 0.7649, and log-loss = 0.6425.