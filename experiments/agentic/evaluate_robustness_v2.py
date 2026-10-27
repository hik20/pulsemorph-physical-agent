"""JSON-native adapter around the unchanged frozen v1 scoring implementation.

Only scalar/container representation is repaired. No thresholds, math, array
values, initial-state rules, or physical acceptance decisions are changed.
"""
from __future__ import annotations
import math
import numpy as np
import evaluate_robustness as _original


def to_json_native(value):
    """Convert NumPy containers/scalars while rejecting all nonfinite numbers."""
    if isinstance(value, np.generic):
        converted = value.item()
        if isinstance(converted, np.generic):
            raise TypeError('NumPy scalar has no lossless native JSON scalar representation')
        return to_json_native(converted)
    if isinstance(value, np.ndarray):
        return to_json_native(value.tolist())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('JSON object keys must be strings')
        return {key:to_json_native(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_native(item) for item in value]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON numeric value')
        return value
    raise TypeError('unsupported JSON value type: ' + type(value).__name__)


def score_features(features, target, tolerance):
    return to_json_native(_original.score_features(features,target,tolerance))


def evaluate_robustness(directory, nominal_baseline, condition_baseline, protocol, goal):
    return to_json_native(_original.evaluate_robustness(
        directory,nominal_baseline,condition_baseline,protocol,goal))
