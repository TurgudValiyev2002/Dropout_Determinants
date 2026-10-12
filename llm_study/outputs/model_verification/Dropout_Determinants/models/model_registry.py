"""Model registry for staged dropout prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory details for one model family."""

    key: str
    display_name: str
    package: str
    class_name: str
    default_params: dict[str, Any]
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    needs_scaling: bool = True
    supports_multiclass: bool = True
    supports_binary: bool = True
    supports_proba: bool = True
    supports_shap: bool = False
    notes: str = ""

    def is_available(self) -> bool:
        try:
            module = import_module(self.package)
            getattr(module, self.class_name)
            return True
        except Exception:
            return False

    def constructor(self) -> Callable[..., Any]:
        try:
            module = import_module(self.package)
            return getattr(module, self.class_name)
        except Exception as exc:
            raise ImportError(
                f"{self.display_name} requires `{self.package}` with `{self.class_name}`."
            ) from exc

    def build(self, **overrides: Any) -> Any:
        params = dict(self.default_params)
        params.update(overrides)
        return self.constructor()(**params)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision_tree": ModelSpec(
        key="decision_tree",
        display_name="Decision Tree",
        package="sklearn.tree",
        class_name="DecisionTreeClassifier",
        default_params={
            "random_state": 123,
            "class_weight": "balanced",
            "max_depth": 8,
            "min_samples_leaf": 10,
        },
        param_grid={
            "max_depth": [3, 5, 8, 12, None],
            "min_samples_leaf": [1, 5, 10, 20],
            "criterion": ["gini", "entropy"],
        },
        supports_shap=True,
        notes="Interpretable nonlinear baseline.",
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        display_name="Random Forest",
        package="sklearn.ensemble",
        class_name="RandomForestClassifier",
        default_params={
            "n_estimators": 500,
            "random_state": 123,
            "class_weight": "balanced",
            "n_jobs": -1,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
        param_grid={
            "n_estimators": [300, 500],
            "max_depth": [None, 8, 12, 16],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", "log2"],
        },
        supports_shap=True,
        notes="Bagging ensemble benchmark for nonlinear interactions.",
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        display_name="XGBoost",
        package="xgboost",
        class_name="XGBClassifier",
        default_params={
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 123,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
        },
        param_grid={
            "n_estimators": [250, 400, 600],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
        supports_shap=True,
        notes="Gradient boosting benchmark and main SHAP candidate.",
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        display_name="LightGBM",
        package="lightgbm",
        class_name="LGBMClassifier",
        default_params={
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 123,
            "class_weight": "balanced",
            "n_jobs": -1,
            "verbose": -1,
        },
        param_grid={
            "n_estimators": [300, 500, 700],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 5, 8],
        },
        supports_shap=True,
        notes="Efficient gradient boosting benchmark.",
    ),
    "catboost": ModelSpec(
        key="catboost",
        display_name="CatBoost",
        package="catboost",
        class_name="CatBoostClassifier",
        default_params={
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "loss_function": "MultiClass",
            "random_seed": 123,
            "verbose": False,
        },
        param_grid={
            "iterations": [300, 500, 700],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
            "l2_leaf_reg": [1, 3, 5],
        },
        supports_shap=True,
        notes="Boosting benchmark; evaluated on the same encoded/scaled matrix for fair comparison.",
    ),
    "rbf_svm": ModelSpec(
        key="rbf_svm",
        display_name="RBF SVM",
        package="sklearn.svm",
        class_name="SVC",
        default_params={
            "C": 1.0,
            "gamma": "scale",
            "kernel": "rbf",
            "probability": True,
            "class_weight": "balanced",
            "random_state": 123,
        },
        param_grid={
            "C": [0.5, 1.0, 3.0, 10.0],
            "gamma": ["scale", 0.01, 0.03, 0.1],
        },
        supports_shap=False,
        notes="Nonlinear margin-based benchmark.",
    ),
    "mlp": ModelSpec(
        key="mlp",
        display_name="Neural Network / MLP",
        package="sklearn.neural_network",
        class_name="MLPClassifier",
        default_params={
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "alpha": 0.0001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "random_state": 123,
        },
        param_grid={
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "activation": ["relu", "tanh"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.0005, 0.001],
        },
        supports_shap=False,
        notes="Neural-network benchmark.",
    ),
}


def available_models() -> dict[str, bool]:
    return {key: spec.is_available() for key, spec in MODEL_REGISTRY.items()}


def get_model_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model key `{key}`. Valid keys: {valid}") from exc


def build_model(key: str, **overrides: Any) -> Any:
    return get_model_spec(key).build(**overrides)


def registry_table() -> list[dict[str, Any]]:
    rows = []
    for key, spec in MODEL_REGISTRY.items():
        rows.append(
            {
                "key": key,
                "model": spec.display_name,
                "available": spec.is_available(),
                "supports_multiclass": spec.supports_multiclass,
                "supports_binary": spec.supports_binary,
                "supports_proba": spec.supports_proba,
                "supports_shap": spec.supports_shap,
                "package": spec.package,
                "class_name": spec.class_name,
                "notes": spec.notes,
            }
        )
    return rows

