# Model Registry Report

This report documents the model families defined in `models/model_registry.py` for the staged early-warning benchmark. All models use the same train/validation/test split and the same transformed input matrix within each stage: one-hot encoded categorical variables and scaled numeric variables.

## Models and Initial Parameters

### 1. Decision Tree

- `max_depth = 8`: limits tree depth to reduce overfitting.
- `min_samples_leaf = 10`: each terminal node must contain at least 10 students.
- `class_weight = "balanced"`: adjusts for class imbalance.
- `criterion`: tuned later between Gini and entropy.

Role in paper: interpretable nonlinear baseline.

### 2. Random Forest

- `n_estimators = 500`: builds 500 trees.
- `min_samples_leaf = 2`: reduces overly specific terminal leaves.
- `max_features = "sqrt"`: each split considers a random subset of features.
- `class_weight = "balanced"`: adjusts for class imbalance.

Role in paper: bagging ensemble benchmark for nonlinear interactions.

### 3. XGBoost

- `n_estimators = 400`: builds 400 boosted trees.
- `learning_rate = 0.05`: conservative boosting step size.
- `max_depth = 4`: controls tree complexity.
- `subsample = 0.9`: each tree uses 90% of rows.
- `colsample_bytree = 0.9`: each tree uses 90% of features.
- `eval_metric = "mlogloss"`: probability-quality objective for multiclass prediction.

Role in paper: main gradient-boosting benchmark and strong SHAP candidate.

### 4. LightGBM

- `n_estimators = 500`: builds 500 boosting iterations.
- `learning_rate = 0.05`: conservative boosting step size.
- `num_leaves = 31`: controls tree complexity.
- `subsample = 0.9`: row sampling.
- `colsample_bytree = 0.9`: feature sampling.
- `class_weight = "balanced"`: adjusts for class imbalance.

Role in paper: efficient gradient-boosting benchmark.

### 5. CatBoost

- `iterations = 500`: builds 500 boosting iterations.
- `learning_rate = 0.05`: conservative step size.
- `depth = 6`: tree depth.
- `loss_function = "MultiClass"`: multiclass classification objective.

Role in paper: boosting benchmark suitable for categorical-heavy tabular problems. For fair comparison, it uses the same encoded/scaled matrix as other models.

### 6. RBF SVM

- `kernel = "rbf"`: nonlinear radial-basis-function kernel.
- `C = 1.0`: regularization parameter.
- `gamma = "scale"`: kernel coefficient based on feature variance.
- `probability = True`: enables probability estimates for log-loss.
- `class_weight = "balanced"`: adjusts for class imbalance.

Role in paper: nonlinear margin-based benchmark.

### 7. Neural Network / MLP

- Hidden layers: `(64, 32)`.
- First hidden layer: 64 neurons.
- Second hidden layer: 32 neurons.
- `activation = "relu"`: hidden-layer nonlinear activation.
- `alpha = 0.0001`: L2 regularization.
- `learning_rate_init = 0.001`: initial learning rate.
- `max_iter = 500`: maximum optimization iterations.
- `early_stopping = True`: stops training if validation performance does not improve.

ReLU is acceptable here because it is used only in hidden layers. The model still performs classification using class probabilities at the output.

## What To Clarify In The Paper

1. All models were evaluated using the same train/validation/test split.

2. Within each timing stage, all models received the same transformed input representation: one-hot encoded categorical variables and scaled numeric variables.

3. Preprocessing was fitted only on the training set and then applied to validation and test sets to avoid leakage.

4. Class imbalance was handled through balanced class weights where supported and through imbalance-aware metrics such as macro-F1 and balanced accuracy.

5. The parameters listed here are initial benchmark settings. Final selected/tuned parameters should be reported separately after validation tuning.

6. CatBoost was kept in the same fair-comparison setup as the other models. Native categorical CatBoost can be tested later as a separate sensitivity analysis.

7. The MLP is included as a neural-network benchmark, not as the main interpretability model. SHAP interpretation should focus primarily on tree-boosting models.
