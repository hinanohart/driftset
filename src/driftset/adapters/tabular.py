# SPDX-License-Identifier: Apache-2.0
"""A column-driven adapter that satisfies :class:`NonconformityAdapter`.

Most scientific-FM calibration tables are flat: one prediction column, one
ground-truth column, some feature columns. ``TabularAdapter`` covers that case
once so the model-specific adapters (Boltz-2 affinity, AlphaFold pLDDT) are thin
configurations rather than re-implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import MissingLabelsError

Transform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TabularAdapter:
    """Map named DataFrame columns onto the adapter protocol.

    Parameters
    ----------
    name:
        Stable identifier used in reports.
    prediction_column:
        Column holding the model point prediction.
    truth_column:
        Column holding the ground-truth label, or ``None`` for inference-only
        frames.
    feature_columns:
        Columns used as covariates for shift detection. Defaults to the
        prediction column alone.
    group_column:
        Optional column providing Mondrian categories.
    prediction_transform / truth_transform:
        Optional element-wise maps applied after extraction (e.g. ``pLDDT/100``
        to put a confidence on the same ``[0, 1]`` scale as lDDT).
    """

    name: str
    prediction_column: str
    truth_column: str | None = None
    feature_columns: tuple[str, ...] = ()
    group_column: str | None = None
    prediction_transform: Transform | None = None
    truth_transform: Transform | None = None

    def _require(self, frame: pd.DataFrame, column: str) -> None:
        if column not in frame.columns:
            raise KeyError(f"adapter {self.name!r}: column {column!r} not in frame")

    def point_predictions(self, frame: pd.DataFrame) -> np.ndarray:
        self._require(frame, self.prediction_column)
        values = frame[self.prediction_column].to_numpy(dtype=float)
        if self.prediction_transform is not None:
            values = np.asarray(self.prediction_transform(values), dtype=float)
        return values

    def true_values(self, frame: pd.DataFrame) -> np.ndarray:
        if self.truth_column is None or self.truth_column not in frame.columns:
            raise MissingLabelsError(
                f"adapter {self.name!r}: no ground-truth column "
                f"{self.truth_column!r} in frame (inference-only data)"
            )
        values = frame[self.truth_column].to_numpy(dtype=float)
        if self.truth_transform is not None:
            values = np.asarray(self.truth_transform(values), dtype=float)
        return values

    def covariate_features(self, frame: pd.DataFrame) -> np.ndarray:
        columns = self.feature_columns or (self.prediction_column,)
        for column in columns:
            self._require(frame, column)
        return frame.loc[:, list(columns)].to_numpy(dtype=float)

    def group_keys(self, frame: pd.DataFrame) -> np.ndarray | None:
        if self.group_column is None:
            return None
        self._require(frame, self.group_column)
        return frame[self.group_column].to_numpy()
