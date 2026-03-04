"""Preprocessing pipeline — composable, per-model data transformations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


class PreprocessingPipeline:
    """Chain of named preprocessing steps executed sequentially."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Callable[[Any], Any]]] = []

    def add_step(self, name: str, function: Callable[[Any], Any]) -> "PreprocessingPipeline":
        """Append a step; returns self for fluent chaining."""
        self._steps.append((name, function))
        return self

    def process(self, raw_input: Any) -> Any:
        """Run all steps in order, returning the final transformed input."""
        data = raw_input
        for step_name, fn in self._steps:
            try:
                data = fn(data)
            except Exception as exc:
                logger.error("preprocessing_step_failed", step=step_name, error=str(exc))
                raise ValueError(f"Preprocessing failed at step '{step_name}': {exc}") from exc
        return data

    @property
    def steps(self) -> list[str]:
        return [name for name, _ in self._steps]

    def __len__(self) -> int:
        return len(self._steps)


# -----------------------------------------------------------------------
# Built-in step functions
# -----------------------------------------------------------------------


def validate_schema(expected_keys: list[str]) -> Callable[[dict], dict]:
    """Return a step that checks all *expected_keys* are present."""

    def _validate(input_data: dict) -> dict:
        missing = [k for k in expected_keys if k not in input_data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return input_data

    return _validate


def normalize_numeric(means: list[float], stds: list[float]) -> Callable[[Any], Any]:
    """Z-score normalisation using pre-computed means & stds."""
    means_arr = np.asarray(means, dtype=np.float64)
    stds_arr = np.asarray(stds, dtype=np.float64)
    stds_arr = np.where(stds_arr == 0, 1.0, stds_arr)

    def _normalize(input_data: Any) -> Any:
        arr = np.asarray(input_data, dtype=np.float64)
        return ((arr - means_arr) / stds_arr).tolist()

    return _normalize


def encode_categorical(mappings: dict[str, dict[str, int]]) -> Callable[[dict], dict]:
    """Map string categories to integer codes.

    *mappings* maps column name → {category_string → integer}.
    """

    def _encode(input_data: dict) -> dict:
        result = dict(input_data)
        for col, mapping in mappings.items():
            if col in result:
                val = result[col]
                if val not in mapping:
                    raise ValueError(f"Unknown category '{val}' for column '{col}'")
                result[col] = mapping[val]
        return result

    return _encode


def handle_missing(strategy: str = "mean", fill_value: float = 0.0) -> Callable[[Any], Any]:
    """Replace NaN / None values.

    Strategies:
        - "mean":  replace NaN with column mean (numpy array expected)
        - "zero":  replace with 0
        - "value": replace with *fill_value*
    """

    def _handle(input_data: Any) -> Any:
        arr = np.asarray(input_data, dtype=np.float64)

        if not np.isnan(arr).any():
            return arr.tolist() if arr.ndim else input_data

        if strategy == "mean":
            if arr.ndim >= 2:
                col_means = np.nanmean(arr, axis=0)
                inds = np.where(np.isnan(arr))
                arr[inds] = col_means[inds[1]] if arr.ndim == 2 else col_means
            else:
                arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
        elif strategy == "zero":
            arr = np.where(np.isnan(arr), 0.0, arr)
        elif strategy == "value":
            arr = np.where(np.isnan(arr), fill_value, arr)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return arr.tolist()

    return _handle


def to_numpy(dtype: str = "float32") -> Callable[[Any], np.ndarray]:
    """Convert input to a numpy array with the given dtype."""

    def _convert(input_data: Any) -> np.ndarray:
        arr = np.asarray(input_data, dtype=dtype)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    return _convert
