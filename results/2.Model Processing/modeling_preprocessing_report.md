# Modeling Preprocessing Report

This stage created the supervised modeling foundation for the dropout determinants study. The original dataset was kept unchanged in `data/data.csv`; all cleaning was done only in memory. Column names were stripped of hidden whitespace, and a unique `StudentID` was added so every student can be tracked consistently across train, validation, and test sets. The dataset contains 4,424 students, no missing values, and no duplicate rows, so no imputation or row deletion was required.

Two target structures were prepared. The first preserves the original multiclass outcome: `Dropout`, `Enrolled`, and `Graduate`. The second creates a binary dropout target: `Dropout` versus `NonDropout`. This is important because the study has two connected aims: predicting full student trajectories and explaining dropout risk specifically.

Feature sets were organized for the next modeling stages. The staged early-warning design includes: Stage A, pre-enrollment variables only; Stage B, pre-enrollment plus first-semester academic performance; and Stage C, pre-enrollment plus first- and second-semester academic performance. This will allow the study to evaluate how early dropout risk can be detected. In addition, ablation feature sets were created to test the added value of academic, financial/demographic, parental, macroeconomic, and course/application variables.

A reproducible stratified train/validation/test split was created with random seed `123`. The final split contains 3,097 students in training, 885 in validation, and 442 in testing. Class proportions are almost identical across all splits: dropout remains about 32.1%, enrolled about 17.9%, and graduate about 49.9%. This ensures that later model comparisons are fair and not affected by different class balances.

Preprocessing metadata was also saved. For the main fair-comparison benchmark, all models should receive the same transformed feature matrix: categorical variables one-hot encoded and numeric variables scaled using parameters fitted only on the training set. This avoids a reviewer concern that models were compared under different input representations. CatBoost native categorical handling can still be tested later as a sensitivity or robustness experiment, but it should not replace the main fair-comparison input.

## One-Hot Encoding and Fair Comparison

Many variables in this dataset are categorical codes, such as course, application mode, previous qualification, nationality, parental qualification, occupation, debtor status, tuition status, scholarship status, and gender. If these codes are passed directly as ordinary numbers, a model may incorrectly treat them as ordered quantities. For example, course code 9991 is not necessarily "larger" or "better" than course code 9130; it is simply a different course. Therefore, for the main fair-comparison models, categorical variables should be transformed using one-hot encoding.

The main benchmark should use the same one-hot encoded categorical variables and the same scaled numeric variables for every model. This makes the comparison fair because XGBoost, LightGBM, CatBoost, Random Forest, SVM, and MLP all receive the same information in the same format. Although CatBoost can handle categorical variables natively, using native CatBoost categories in the main table while other models use one-hot encoding would make the input representation different. Native CatBoost can still be reported separately as a sensitivity test.

## What To State In The Paper

1. The analysis used the full SATDAP dataset of 4,424 students with no missing observations and no duplicate records.

2. The outcome was modeled in two ways: a three-class student-status outcome (`Dropout`, `Enrolled`, `Graduate`) and a binary dropout-risk outcome (`Dropout` vs. `NonDropout`) for dropout-specific interpretation.

3. The supervised prediction models used a fixed stratified train/validation/test split of 70%/20%/10%, preserving the original class proportions in each subset.

4. The early-warning design was organized into three timing stages: pre-enrollment information only, pre-enrollment plus first-semester performance, and pre-enrollment plus first- and second-semester performance.

5. Ablation models were planned to estimate the incremental value of different information blocks: academic, financial/demographic, parental background, macroeconomic, and course/application variables.

6. For the main model comparison, all models used the same input representation: one-hot encoded categorical variables and scaled numeric variables, with preprocessing fitted only on the training set and then applied to validation/test data.

7. The graph and risk-clustering analyses should be described separately as exploratory full-dataset analyses, not as supervised train/test prediction exercises.

