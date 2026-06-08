# Feature Importance and SHAP Interpretation Report

This stage explains the tuned models selected by validation performance: Stage A, Stage B, Stage C, the full-feature model, and the academic + demographic/financial ablation model. Feature importance is reported in two complementary ways: permutation importance on the held-out test set and SHAP importance for the tuned LightGBM models.

Top-10 features are recommended for the paper instead of top-20 because they are easier to read and more defensible in a limited manuscript table or figure. Full feature tables are still saved for auditability.

## Model-Level Summary

- Full feature tuned (LightGBM): test Macro-F1 0.7016, balanced accuracy 0.7892, log-loss 0.5902. Top dropout SHAP feature: Curricular units 2nd sem (approved). Top permutation feature: Curricular units 2nd sem (approved).
- Academic + demographic-financial tuned (LightGBM): test Macro-F1 0.6983, balanced accuracy 0.7851, log-loss 0.6213. Top dropout SHAP feature: Curricular units 2nd sem (approved). Top permutation feature: Curricular units 2nd sem (approved).
- Stage C tuned (LightGBM): test Macro-F1 0.6981, balanced accuracy 0.7839, log-loss 0.5971. Top dropout SHAP feature: Curricular units 2nd sem (approved). Top permutation feature: Curricular units 2nd sem (approved).
- Stage B tuned (LightGBM): test Macro-F1 0.6802, balanced accuracy 0.7728, log-loss 0.6467. Top dropout SHAP feature: Curricular units 1st sem (approved). Top permutation feature: Curricular units 1st sem (approved).
- Stage A tuned (LightGBM): test Macro-F1 0.5713, balanced accuracy 0.6885, log-loss 0.8492. Top dropout SHAP feature: Tuition fees up to date. Top permutation feature: Tuition fees up to date.

## Policy Interpretation

The main paper interpretation should emphasize variables that appear repeatedly across the Stage C, full-feature, and academic + demographic/financial models. Repeated features are more reliable than one-model-only signals. The strongest common dropout-risk structure is expected around semester performance, tuition/financial vulnerability, and student profile variables.

Common top-10 dropout SHAP features across Stage C, full-feature, and academic + demographic/financial models: Age at enrollment, Curricular units 1st sem (approved), Curricular units 2nd sem (approved), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (grade), Tuition fees up to date.

Use the dropout-class SHAP plots when discussing dropout risk specifically. Use the global SHAP plots when discussing overall multiclass model behavior. Use permutation importance as a robustness check because it measures held-out performance degradation when each raw feature is disrupted.