"""
Feature Engineering Module.

Transformations derived from EDA_FE.ipynb for preparing data
for model training.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config import COLUMN_RENAME_MAP

logger = logging.getLogger(__name__)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns from raw names to clean standardized names.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with renamed columns
    """
    rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed {len(rename_map)} columns")
    return df


def parse_activation_date(
    df: pd.DataFrame,
    date_col: str = 'activation_date',
) -> pd.DataFrame:
    """
    Parse activation date column from various formats to datetime.

    Handles format where date is stored as integer with last 6 digits as YYYYMM.

    Args:
        df: Input DataFrame
        date_col: Name of date column

    Returns:
        DataFrame with parsed date column
    """
    df = df.copy()

    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found")
        return df

    # Convert to string and extract last 6 digits (YYYYMM format)
    df[date_col] = df[date_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col].str[-6:], format='%Y%m')

    logger.info(f"Parsed {date_col} to datetime")
    return df


def create_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flag features from count/continuous variables.

    Transforms:
    - maxarrs_l12m → maxarrs_l12m_flag (>0)
    - total_payment_reversals → total_payment_reversals_flag (>0)
    - monthssincemrrdpayment → monthssincemrrdpayment_flag (>=0)
    - timesrdpay_l6m → timesrdpay_l6m_flag (>0)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with new binary flag columns (originals dropped)
    """
    df = df.copy()

    # maxarrs_l12m → flag
    if 'maxarrs_l12m' in df.columns:
        df['maxarrs_l12m_flag'] = (df['maxarrs_l12m'] > 0).astype('int8')
        df = df.drop(columns=['maxarrs_l12m'])
        logger.info("Created maxarrs_l12m_flag")

    # total_payment_reversals → flag
    if 'total_payment_reversals' in df.columns:
        df['total_payment_reversals_flag'] = (
            df['total_payment_reversals'].fillna(-1) > 0
        ).astype('int8')
        df = df.drop(columns=['total_payment_reversals'])
        logger.info("Created total_payment_reversals_flag")

    # monthssincemrrdpayment → flag
    if 'monthssincemrrdpayment' in df.columns:
        df['monthssincemrrdpayment_flag'] = (
            df['monthssincemrrdpayment'] >= 0
        ).astype('int8')
        df = df.drop(columns=['monthssincemrrdpayment'])
        logger.info("Created monthssincemrrdpayment_flag")

    # timesrdpay_l6m → flag
    if 'timesrdpay_l6m' in df.columns:
        df['timesrdpay_l6m_flag'] = (
            df['timesrdpay_l6m'].fillna(-1) > 0
        ).astype('int8')
        df = df.drop(columns=['timesrdpay_l6m'])
        logger.info("Created timesrdpay_l6m_flag")

    return df


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'mode',
) -> pd.DataFrame:
    """
    Handle missing values in DataFrame.

    Args:
        df: Input DataFrame
        numeric_strategy: Strategy for numeric columns ('median', 'mean', 'zero')
        categorical_strategy: Strategy for categorical columns ('mode', 'missing')

    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()

    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if df[col].isna().any():
            if numeric_strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == 'zero':
                df[col] = df[col].fillna(0)

    # Handle categorical columns
    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    for col in cat_cols:
        if df[col].isna().any():
            if categorical_strategy == 'mode':
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
            elif categorical_strategy == 'missing':
                if df[col].dtype.name == 'category':
                    df[col] = df[col].cat.add_categories('Missing')
                df[col] = df[col].fillna('Missing')

    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} missing values remain after imputation")
    else:
        logger.info("All missing values handled")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    target_col: str,
    high_cardinality_threshold: int = 10,
) -> pd.DataFrame:
    """
    Encode categorical variables for model training.

    Strategy:
    - Low cardinality (≤ threshold): One-hot encoding with drop_first=True
    - High cardinality (> threshold): Frequency encoding

    Args:
        df: Input DataFrame
        target_col: Target column name (to exclude from encoding)
        high_cardinality_threshold: Threshold for encoding strategy selection

    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()

    cat_cols = df.select_dtypes(include=['category', 'object']).columns
    cat_cols = [c for c in cat_cols if c != target_col]

    for col in cat_cols:
        n_unique = df[col].nunique()

        if n_unique <= high_cardinality_threshold:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            logger.debug(f"One-hot encoded {col} ({n_unique} categories)")
        else:
            # Frequency encoding
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)
            df = df.rename(columns={col: f"{col}_freq"})
            logger.debug(f"Frequency encoded {col} ({n_unique} categories)")

    logger.info(f"Encoded {len(cat_cols)} categorical columns")
    return df


def preprocess_data(
    df: pd.DataFrame,
    target_col: str = 'bad',
    date_col: Optional[str] = 'activation_date',
    encode_cats: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Applies all transformations in sequence:
    1. Parse activation date
    2. Create binary flags
    3. Handle missing values
    4. Encode categoricals (optional)

    Args:
        df: Raw input DataFrame
        target_col: Target column name
        date_col: Date column name (or None to skip)
        encode_cats: Whether to encode categorical variables

    Returns:
        Preprocessed DataFrame ready for splitting
    """
    logger.info("Starting preprocessing pipeline")

    # Parse dates
    if date_col and date_col in df.columns:
        df = parse_activation_date(df, date_col)

    # Create binary flags
    df = create_binary_flags(df)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categoricals
    if encode_cats:
        df = encode_categoricals(df, target_col)

    logger.info(
        f"Preprocessing complete: {len(df):,} rows, {len(df.columns)} columns"
    )
    return df
