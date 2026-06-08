# Hyperparameter Tuning Report

This tuning step selected the best parameter setting for each top model family within each early-warning stage using validation Macro-F1 as the main criterion.

## Best Tuned Model By Stage
- A_pre_enrollment: LightGBM with validation Macro-F1=0.5831, Balanced Accuracy=0.7001, Log-Loss=0.8491.
- B_first_semester: LightGBM with validation Macro-F1=0.6793, Balanced Accuracy=0.7715, Log-Loss=0.6591.
- C_second_semester: LightGBM with validation Macro-F1=0.7120, Balanced Accuracy=0.7969, Log-Loss=0.5947.

## Main Results In Simple Terms

Hyperparameter tuning confirms the main early-warning finding: prediction improves strongly when academic performance information is added. The best validation Macro-F1 increases from 0.5831 in the pre-enrollment stage to 0.6793 after first-semester information and 0.7120 after second-semester information. This means pre-enrollment data can identify some risk, but first-semester academic performance is the first major improvement point. Second-semester information improves prediction further.

After tuning, LightGBM is the strongest validation model in all three stages. Its best Stage C validation results are Macro-F1 = 0.7120, balanced accuracy = 0.7969, and log-loss = 0.5947. This is a meaningful improvement over the untuned benchmark and gives a stronger final candidate for the staged early-warning model.

The test results are also useful. In Stage C, RBF SVM has the highest test Macro-F1 = 0.7091, while the validation-selected LightGBM has test Macro-F1 = 0.6981. We should not select the final model based on test performance, because the test set must remain an unbiased final check. Therefore, LightGBM should be treated as the validation-selected model, while the strong RBF SVM test result should be discussed as evidence that several nonlinear models perform competitively after second-semester information is included.

Overfitting is still important. Some models have high train performance and lower validation/test performance. The overfitting gap table should be used to show that we considered model stability, not only the highest metric. Random Forest and LightGBM remain strong, but the final paper should present train, validation, and test metrics together so reviewers can see generalization behavior.

## Recommended Paper Message

The tuned benchmark supports a staged Early Warning System. Universities can identify some dropout risk using pre-enrollment information, but predictive performance improves substantially after first-semester academic results become available. This suggests that first-semester monitoring is a key intervention window. Second-semester information adds additional predictive power, but the first large gain appears after the first semester.

## Main Outputs
- `csv/tuning_candidate_validation_results.csv`
- `csv/tuned_model_comparison.csv`
- `csv/best_tuned_validation_model_by_stage.csv`
- `csv/tuned_overfitting_gap_table.csv`
- `plots/`
