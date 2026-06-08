# Ablation Study Summary Report

This ablation study compares whether additional variable blocks improve dropout prediction beyond academic performance. Each block was trained with the same train/validation/test split and the same encoded/scaled input protocol.

## Best Tuned Validation Models

- 2.Academic + demographic-financial: LightGBM | validation Macro-F1 0.7022; test Macro-F1 0.6983; test log-loss 0.6213.
- 4.Academic + macroeconomic: Random Forest | validation Macro-F1 0.6731; test Macro-F1 0.6584; test log-loss 0.6561.
- 5.Academic + course-application: LightGBM | validation Macro-F1 0.6718; test Macro-F1 0.6546; test log-loss 0.6417.
- 3.Academic + parental education: LightGBM | validation Macro-F1 0.6653; test Macro-F1 0.6695; test log-loss 0.6425.
- 1.Academic only: Random Forest | validation Macro-F1 0.6649; test Macro-F1 0.6534; test log-loss 0.6638.

## Paper Interpretation

The strongest ablation block on the held-out test set was **2.Academic + demographic-financial**, selected by validation performance and tested independently. This tells us which information block adds the most practical value after academic signals.

For the paper, use the `paper_ready_best_tuned_ablation_table.csv` table and the `best_tuned_ablation_blocks_test_macro_f1.png` plot. These directly answer whether demographic/financial, parental, macroeconomic, or course/application information improves prediction after academic performance.