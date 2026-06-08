# Staged Early Warning Results: One-Page Report

The staged early-warning benchmark is now correctly structured. It compares seven model families, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, RBF SVM, and Neural Network/MLP, across three information stages. Each model is evaluated on the same train/validation/test split and receives the same transformed input representation within each stage. The final result table contains 63 rows, which is correct: 3 stages x 7 models x 3 splits.

The main finding is clear: predictive performance improves as student academic information becomes available. In Stage A, using only pre-enrollment information, the best validation model is Random Forest with Macro-F1 = 0.5787, balanced accuracy = 0.6930, and log-loss = 0.8539. This shows that background, financial, course, and admission information already contains meaningful risk signals, but prediction remains limited before academic performance is observed.

In Stage B, after adding first-semester academic indicators, performance improves substantially. The best validation model is Neural Network/MLP with Macro-F1 = 0.6689, balanced accuracy = 0.7564, and log-loss = 0.6769. This is a strong early-warning result: the first semester adds major predictive value and should be treated as a critical intervention window for universities.

In Stage C, after adding second-semester academic indicators, performance improves again. The best validation model remains Neural Network/MLP with Macro-F1 = 0.6995, balanced accuracy = 0.7807, and log-loss = 0.5829. Random Forest is very close in Stage C, with validation Macro-F1 = 0.6946 and balanced accuracy = 0.7807, while XGBoost also performs strongly with Macro-F1 = 0.6954 and the lowest validation log-loss among the strongest models at 0.5692. This means the final model choice should consider not only Macro-F1 but also probability quality and interpretability.

The test results confirm the same general pattern. Stage A is weakest, Stage B improves clearly, and Stage C is strongest. In Stage C test results, Random Forest has the highest Macro-F1 = 0.6998 and balanced accuracy = 0.7844. RBF SVM and Neural Network/MLP are also competitive, while XGBoost and CatBoost have strong log-loss values. This supports the paper’s core argument: dropout prediction becomes much stronger after universities observe actual academic progression.

One important modeling caution is overfitting. LightGBM reaches perfect or near-perfect training performance in later stages but does not dominate validation/test performance. This suggests overfitting under the current default settings. Random Forest also has high training performance, but its validation/test results are more stable. Therefore, the current results should be described as an initial benchmark using fixed starting parameters. The next modeling step should tune the strongest candidates, especially Random Forest, XGBoost, CatBoost, RBF SVM, and MLP.

For the paper, the most useful result is not simply “one model is best.” The stronger message is that early-warning timing matters. Pre-enrollment information gives a baseline risk profile, first-semester performance provides a large improvement, and second-semester performance improves the prediction further. This directly supports a staged Early Warning System for universities.

Recommended paper outputs from this stage:

1. A table with train, validation, and test Macro-F1, balanced accuracy, and log-loss for all models and stages.
2. A compact “best model by stage” table.
3. A line plot of validation Macro-F1 across Stage A, B, and C.
4. A line plot of validation log-loss across Stage A, B, and C.
5. A Stage C model-ranking plot using test Macro-F1 or validation Macro-F1.

Conclusion: the staged modeling is now correctly organized and produces a strong research result. The findings support the idea that universities can detect some dropout risk before enrollment, but the first semester is the key moment where prediction becomes much more reliable.
