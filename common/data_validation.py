"""
Data validation utilities for ML pipelines.

This module provides comprehensive data validation functions including:
- Input validation with type checking
- Schema validation against expected formats
- Missing value detection and reporting
- Outlier detection using statistical methods

Usage:
    from common.data_validation import validate_input, validate_schema

    # Validate input types
    @validate_input
    def process_data(text: str, score: float) -> dict:
        return {"text": text, "score": score}

    # Validate against schema
    schema = {"text": str, "score": float, "label": int}
    validate_schema(data, schema)
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_input(func: Callable) -> Callable:
    """
    Decorator to validate function inputs match their type hints.

    Args:
        func: Function to decorate with input validation

    Returns:
        Wrapped function with validation

    Raises:
        ValidationError: If input types don't match annotations

    Example:
        @validate_input
        def process(text: str, count: int) -> str:
            return text * count
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function annotations
        annotations = func.__annotations__.copy()
        annotations.pop('return', None)

        # Build arg name to value mapping
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        arg_dict = dict(zip(arg_names, args))
        arg_dict.update(kwargs)

        # Validate each argument
        for arg_name, expected_type in annotations.items():
            if arg_name in arg_dict:
                value = arg_dict[arg_name]
                if value is not None and not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Argument '{arg_name}' expected {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        return func(*args, **kwargs)

    return wrapper


def validate_schema(
    data: Union[Dict[str, Any], pd.DataFrame],
    schema: Dict[str, Type],
    strict: bool = True
) -> bool:
    """
    Validate data against expected schema.

    Args:
        data: Data to validate (dict or DataFrame)
        schema: Expected schema as {field_name: expected_type}
        strict: If True, raise error on validation failure. If False, return bool

    Returns:
        True if validation passes, False otherwise (when strict=False)

    Raises:
        ValidationError: If validation fails and strict=True

    Example:
        schema = {"user_id": int, "name": str, "score": float}
        validate_schema({"user_id": 1, "name": "Alice", "score": 0.9}, schema)
    """
    # Convert DataFrame to dict for validation
    if isinstance(data, pd.DataFrame):
        if len(data) == 0:
            if strict:
                raise ValidationError("DataFrame is empty")
            return False
        data_dict = data.iloc[0].to_dict()
    else:
        data_dict = data

    errors = []

    # Check for missing required fields
    missing_fields = set(schema.keys()) - set(data_dict.keys())
    if missing_fields:
        errors.append(f"Missing required fields: {missing_fields}")

    # Check types for existing fields
    for field, expected_type in schema.items():
        if field in data_dict:
            value = data_dict[field]
            if value is not None and not isinstance(value, expected_type):
                # Handle numpy types
                if expected_type == float and isinstance(value, (int, np.integer, np.floating)):
                    continue
                if expected_type == int and isinstance(value, (np.integer,)):
                    continue

                errors.append(
                    f"Field '{field}' expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                )

    if errors:
        if strict:
            raise ValidationError("; ".join(errors))
        return False

    return True


def check_missing_values(
    data: pd.DataFrame,
    threshold: float = 0.5,
    raise_error: bool = False
) -> Dict[str, float]:
    """
    Check for missing values in DataFrame columns.

    Args:
        data: DataFrame to check
        threshold: Maximum allowed missing ratio (0 to 1)
        raise_error: If True, raise error when threshold exceeded

    Returns:
        Dict mapping column names to missing value ratios

    Raises:
        ValidationError: If missing ratio exceeds threshold and raise_error=True

    Example:
        missing_info = check_missing_values(df, threshold=0.3)
        # Returns: {"col1": 0.1, "col2": 0.25, "col3": 0.4}
    """
    if data.empty:
        logger.warning("Empty DataFrame provided to check_missing_values")
        return {}

    missing_ratios = {}
    columns_exceeding_threshold = []

    for column in data.columns:
        missing_count = data[column].isna().sum()
        missing_ratio = missing_count / len(data)

        if missing_ratio > 0:
            missing_ratios[column] = missing_ratio

            if missing_ratio > threshold:
                columns_exceeding_threshold.append(
                    f"{column} ({missing_ratio:.1%})"
                )

    if columns_exceeding_threshold:
        message = (
            f"Columns exceeding missing value threshold ({threshold:.1%}): "
            f"{', '.join(columns_exceeding_threshold)}"
        )

        if raise_error:
            raise ValidationError(message)
        else:
            warnings.warn(message)

    return missing_ratios


def detect_outliers(
    data: Union[pd.Series, np.ndarray, List[float]],
    method: str = "iqr",
    threshold: float = 1.5
) -> np.ndarray:
    """
    Detect outliers in numerical data using statistical methods.

    Args:
        data: Numerical data to check for outliers
        method: Detection method - "iqr" (Interquartile Range) or "zscore"
        threshold: Sensitivity threshold (1.5 for IQR, 3.0 for z-score typical)

    Returns:
        Boolean array indicating outlier positions (True = outlier)

    Raises:
        ValidationError: If invalid method specified

    Example:
        outliers = detect_outliers(df["price"], method="iqr", threshold=1.5)
        clean_data = df[~outliers]
    """
    # Convert to numpy array
    if isinstance(data, pd.Series):
        arr = data.values
    elif isinstance(data, list):
        arr = np.array(data)
    else:
        arr = data

    # Remove NaN values for calculation
    arr_clean = arr[~np.isnan(arr)]

    if len(arr_clean) == 0:
        logger.warning("All values are NaN in outlier detection")
        return np.zeros(len(arr), dtype=bool)

    if method == "iqr":
        # Interquartile Range method
        q1 = np.percentile(arr_clean, 25)
        q3 = np.percentile(arr_clean, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = (arr < lower_bound) | (arr > upper_bound)

    elif method == "zscore":
        # Z-score method
        mean = np.mean(arr_clean)
        std = np.std(arr_clean)

        if std == 0:
            logger.warning("Standard deviation is 0, no outliers detected")
            return np.zeros(len(arr), dtype=bool)

        z_scores = np.abs((arr - mean) / std)
        outliers = z_scores > threshold

    else:
        raise ValidationError(
            f"Invalid outlier detection method: {method}. "
            f"Use 'iqr' or 'zscore'"
        )

    outlier_count = outliers.sum()
    outlier_ratio = outlier_count / len(arr)

    logger.info(
        f"Detected {outlier_count} outliers ({outlier_ratio:.1%}) "
        f"using {method} method with threshold={threshold}"
    )

    return outliers


def validate_numeric_range(
    value: Union[int, float],
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    inclusive: bool = True
) -> bool:
    """
    Validate that a numeric value falls within specified range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value (None for no minimum)
        max_value: Maximum allowed value (None for no maximum)
        inclusive: If True, use <= and >=. If False, use < and >

    Returns:
        True if value is within range, False otherwise

    Example:
        validate_numeric_range(5.5, min_value=0.0, max_value=10.0)  # True
        validate_numeric_range(15, min_value=0, max_value=10)  # False
    """
    if min_value is not None:
        if inclusive and value < min_value:
            return False
        if not inclusive and value <= min_value:
            return False

    if max_value is not None:
        if inclusive and value > max_value:
            return False
        if not inclusive and value >= max_value:
            return False

    return True


def validate_dataframe_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    optional_columns: Optional[List[str]] = None
) -> None:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        optional_columns: List of column names that may be present

    Raises:
        ValidationError: If required columns are missing

    Example:
        validate_dataframe_columns(
            df,
            required_columns=["user_id", "timestamp"],
            optional_columns=["metadata"]
        )
    """
    df_columns = set(df.columns)
    required_set = set(required_columns)

    missing_columns = required_set - df_columns
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if optional_columns:
        optional_set = set(optional_columns)
        allowed_columns = required_set | optional_set
        extra_columns = df_columns - allowed_columns

        if extra_columns:
            warnings.warn(
                f"DataFrame contains unexpected columns: {sorted(extra_columns)}"
            )

    logger.info(f"DataFrame validation passed for {len(required_columns)} required columns")
