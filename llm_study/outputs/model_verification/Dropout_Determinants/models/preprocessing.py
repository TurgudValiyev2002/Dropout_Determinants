"""Preprocessing helpers for dropout prediction experiments.

The main benchmark uses one common input representation for every model:
one-hot encoded categorical variables plus scaled numeric variables. The
preprocessing itself is lightweight pandas/numpy work and is fitted only on
the training set before being applied to validation and test sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TabularOneHotScaler:
    """Fit train-only one-hot/scaling metadata and transform tabular data."""

    numeric_features: list[str]
    categorical_features: list[str]
    numeric_means_: dict[str, float] = field(default_factory=dict)
    numeric_stds_: dict[str, float] = field(default_factory=dict)
    categories_: dict[str, list[str]] = field(default_factory=dict)
    output_columns_: list[str] = field(default_factory=list)

    def fit(self, data: pd.DataFrame) -> "TabularOneHotScaler":
        """Fit preprocessing metadata using training data only."""
        self.numeric_means_ = {}
        self.numeric_stds_ = {}
        for col in self.numeric_features:
            values = pd.to_numeric(data[col], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            self.numeric_means_[col] = mean
            self.numeric_stds_[col] = std if std > 0 else 1.0

        self.categories_ = {}
        for col in self.categorical_features:
            values = data[col].astype("string").fillna("__MISSING__")
            self.categories_[col] = sorted(values.unique().tolist())

        transformed = self.transform(data, allow_unfitted_output=True)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, data: pd.DataFrame, allow_unfitted_output: bool = False) -> pd.DataFrame:
        """Transform data using fitted train-only metadata."""
        numeric_parts = []
        for col in self.numeric_features:
            values = pd.to_numeric(data[col], errors="coerce").fillna(self.numeric_means_[col])
            scaled = (values - self.numeric_means_[col]) / self.numeric_stds_[col]
            numeric_parts.append(scaled.rename(col))

        parts = []
        if numeric_parts:
            parts.append(pd.concat(numeric_parts, axis=1))

        for col in self.categorical_features:
            values = data[col].astype("string").fillna("__MISSING__")
            cat_type = pd.CategoricalDtype(categories=self.categories_[col])
            dummies = pd.get_dummies(values.astype(cat_type), prefix=col, dtype=float)
            parts.append(dummies)

        transformed = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=data.index)

        if not allow_unfitted_output:
            transformed = transformed.reindex(columns=self.output_columns_, fill_value=0.0)

        return transformed.astype("float32")

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit on data and return transformed matrix."""
        return self.fit(data).transform(data)

    def transform_numpy(self, data: pd.DataFrame) -> np.ndarray:
        """Transform data and return a float32 numpy array."""
        return self.transform(data).to_numpy(dtype=np.float32)
