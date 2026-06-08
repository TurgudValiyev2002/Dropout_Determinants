# Before-and-After Hyperparameter Tuning Report

This report compares the initial staged-model benchmark with the tuned benchmark. The comparison uses the same dataset split, the same stages, and the same feature representation, so differences reflect parameter changes rather than different data or variables.

## What Was Tuned
- Random Forest: 8 candidate parameter settings.
- XGBoost: 8 candidate parameter settings.
- LightGBM: 16 candidate parameter settings.
- CatBoost: 8 candidate parameter settings.
- RBF SVM: 6 candidate parameter settings.
- Neural Network / MLP: 12 candidate parameter settings.

In total, 174 candidate model-stage combinations were evaluated during tuning. The final tuned comparison contains 54 rows: 3 stages x 6 tuned models x 3 splits.

## Main Tuning Result
- A_pre_enrollment: LightGBM selected by validation Macro-F1. Validation Macro-F1=0.5831, balanced accuracy=0.7001, log-loss=0.8491.
- B_first_semester: LightGBM selected by validation Macro-F1. Validation Macro-F1=0.6793, balanced accuracy=0.7715, log-loss=0.6591.
- C_second_semester: LightGBM selected by validation Macro-F1. Validation Macro-F1=0.7120, balanced accuracy=0.7969, log-loss=0.5947.

## Validation-Set Changes
### A_pre_enrollment
- LightGBM: Macro-F1 0.5336 -> 0.5831 (+0.0495); balanced accuracy 0.6590 -> 0.7001 (+0.0412); log-loss 0.9615 -> 0.8491 (improved by 0.1124).
- RBF SVM: Macro-F1 0.5547 -> 0.5788 (+0.0241); balanced accuracy 0.6792 -> 0.6970 (+0.0178); log-loss 0.8122 -> 0.8106 (improved by 0.0016).
- Neural Network / MLP: Macro-F1 0.5145 -> 0.5344 (+0.0199); balanced accuracy 0.6578 -> 0.6627 (+0.0049); log-loss 0.8113 -> 0.8337 (worsened by 0.0223).
- Random Forest: Macro-F1 0.5787 -> 0.5795 (+0.0008); balanced accuracy 0.6930 -> 0.6939 (+0.0009); log-loss 0.8539 -> 0.8547 (worsened by 0.0008).
- XGBoost: Macro-F1 0.5489 -> 0.5489 (+0.0000); balanced accuracy 0.6739 -> 0.6739 (+0.0000); log-loss 0.7983 -> 0.7983 (worsened by 0.0000).
- CatBoost: Macro-F1 0.5145 -> 0.5145 (+0.0000); balanced accuracy 0.6573 -> 0.6573 (+0.0000); log-loss 0.7993 -> 0.7993 (worsened by 0.0000).
### B_first_semester
- LightGBM: Macro-F1 0.6672 -> 0.6793 (+0.0122); balanced accuracy 0.7590 -> 0.7715 (+0.0126); log-loss 0.7832 -> 0.6591 (improved by 0.1241).
- RBF SVM: Macro-F1 0.6568 -> 0.6651 (+0.0084); balanced accuracy 0.7560 -> 0.7630 (+0.0069); log-loss 0.6628 -> 0.6615 (improved by 0.0013).
- Random Forest: Macro-F1 0.6622 -> 0.6662 (+0.0040); balanced accuracy 0.7550 -> 0.7608 (+0.0058); log-loss 0.7097 -> 0.7432 (worsened by 0.0335).
- XGBoost: Macro-F1 0.6598 -> 0.6598 (+0.0000); balanced accuracy 0.7542 -> 0.7542 (+0.0000); log-loss 0.6302 -> 0.6302 (worsened by 0.0000).
- CatBoost: Macro-F1 0.6373 -> 0.6373 (+0.0000); balanced accuracy 0.7413 -> 0.7413 (+0.0000); log-loss 0.6387 -> 0.6387 (worsened by 0.0000).
- Neural Network / MLP: Macro-F1 0.6689 -> 0.6689 (+0.0000); balanced accuracy 0.7564 -> 0.7564 (+0.0000); log-loss 0.6769 -> 0.6769 (worsened by 0.0000).
### C_second_semester
- LightGBM: Macro-F1 0.6831 -> 0.7120 (+0.0289); balanced accuracy 0.7717 -> 0.7969 (+0.0253); log-loss 0.7272 -> 0.5947 (improved by 0.1325).
- Random Forest: Macro-F1 0.6946 -> 0.7010 (+0.0064); balanced accuracy 0.7807 -> 0.7844 (+0.0038); log-loss 0.6387 -> 0.6397 (worsened by 0.0010).
- RBF SVM: Macro-F1 0.6755 -> 0.6803 (+0.0049); balanced accuracy 0.7700 -> 0.7733 (+0.0032); log-loss 0.6039 -> 0.6047 (worsened by 0.0008).
- XGBoost: Macro-F1 0.6954 -> 0.7001 (+0.0047); balanced accuracy 0.7773 -> 0.7799 (+0.0026); log-loss 0.5692 -> 0.5718 (worsened by 0.0026).
- CatBoost: Macro-F1 0.6788 -> 0.6788 (+0.0000); balanced accuracy 0.7666 -> 0.7666 (+0.0000); log-loss 0.5764 -> 0.5764 (worsened by 0.0000).
- Neural Network / MLP: Macro-F1 0.6995 -> 0.6995 (+0.0000); balanced accuracy 0.7807 -> 0.7807 (+0.0000); log-loss 0.5829 -> 0.5829 (worsened by 0.0000).

## Test-Set Check
- A_pre_enrollment: best tuned test Macro-F1 is RBF SVM (0.5818); best tuned test log-loss is CatBoost (0.7999).
- B_first_semester: best tuned test Macro-F1 is LightGBM (0.6802); best tuned test log-loss is RBF SVM (0.5949).
- C_second_semester: best tuned test Macro-F1 is RBF SVM (0.7091); best tuned test log-loss is RBF SVM (0.5432).

## Interpretation For The Paper
Hyperparameter tuning is important because it improves the reliability of the staged model comparison and reduces the chance that conclusions are driven by arbitrary default settings. The biggest paper-level result remains the staged early-warning pattern: Stage B improves strongly over Stage A, and Stage C is strongest overall. Tuning strengthens this result because the validation-selected best model improves from Stage A Macro-F1 0.5831 to Stage B 0.6793 and Stage C 0.7120.

Tuning did not improve every model equally. Some models gained only modestly, and in a few cases validation log-loss became worse while Macro-F1 improved. This matters because Macro-F1 measures class-label quality, while log-loss measures probability quality. For a university early-warning system, both are useful: Macro-F1 tells us whether the model separates dropout/enrolled/graduate groups, while log-loss tells us whether predicted risks are well calibrated.

For model selection, the validation set should be used. The test set should be presented only as the final generalization check. In the tuned results, LightGBM is selected by validation Macro-F1 in all three stages, while some other models, especially RBF SVM and Random Forest, are competitive on the test set. This should be reported carefully as evidence that several nonlinear models work well after academic performance variables are included.

## Recommended Tables/Figures
- Table: before-vs-after validation Macro-F1, balanced accuracy, and log-loss by model and stage.
- Table: tuned train/validation/test metrics for all models and stages.
- Figure: tuned validation Macro-F1 across Stage A, B, C.
- Figure: tuned validation log-loss across Stage A, B, C.
- Appendix table: best parameter setting for each model and stage.