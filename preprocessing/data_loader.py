"""
Data Loading Module.

Functions for loading raw CSV data, converting column types,
and creating temporal train/validation/test splits.

Author: Lebohang Andile Skungwini
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import (
    RANDOM_SEED,
    COLUMN_CONFIG,
    COLUMNS_TO_DROP,
    COLUMN_RENAME_MAP,
)

logger = logging.getLogger(__name__)


def load_raw_data(
    path: Union[str, Path],
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Load raw CSV data with basic cleaning.

    Args:
        path: Path to CSV file
        drop_duplicates: Whether to drop duplicate rows

    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {path}")

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    if drop_duplicates:
        initial_len = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_len:
            logger.info(f"Removed {initial_len - len(df):,} duplicate rows")

    # Drop predefined columns if they exist
    cols_to_drop = [c for c in COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")

    # Rename columns
    rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Renamed {len(rename_map)} columns")

    return df


def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.

    Converts categorical columns to 'category', binary flags to 'int8',
    and integer columns to appropriate integer types.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with converted types
    """
    df = df.copy()

    # Define columns by type
    category_columns = [
        'client_type', 'contract_type', 'financed_deal_flag',
        'sales_region', 'approved_with_condition_flag', 'conditions_met',
        'portfolio', 'Channel', 'Term', 'Phones', 'Bank',
        'score_band', 'income_band', 'customer_value',
        'internal_risk_ranking', 'times0_l3m', 'times0_l6m',
    ]

    int8_columns = [
        'bad', 'returned_debit_flag', 'thinfile',
        'maxarrs_l12m_flag', 'total_payment_reversals_flag',
        'monthssincemrrdpayment_flag', 'timesrdpay_l6m_flag',
    ]

    int32_columns = [
        'months_since_acc_creation', 'maxarrs_l12m',
        'total_payment_reversals', 'times0_l6m',
    ]

    # Convert category columns
    for col in category_columns:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # Convert int8 columns
    for col in int8_columns:
        if col in df.columns:
            df[col] = df[col].astype('int8')

    # Convert int32 columns
    for col in int32_columns:
        if col in df.columns:
            df[col] = df[col].astype('Int32')

    logger.info("Column types converted")
    return df


def temporal_split(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    train_months: List[int],
    val_months: List[int],
    test_months_early: List[int],
    test_months_late: List[int],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data based on temporal windows for out-of-time validation.

    Args:
        df: DataFrame with features and target
        date_col: Column containing activation dates
        target_col: Target column name
        train_months: Month numbers for training (e.g., 1-18)
        val_months: Month numbers for validation (e.g., 19-21)
        test_months_early: Month numbers for early OOT test (e.g., 22-23)
        test_months_late: Month numbers for late OOT test (e.g., 24)

    Returns:
        Dictionary with 'train', 'val', 'test_early', 'test_late' tuples of (X, y)
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    # Create month index (1-indexed)
    min_date = df[date_col].min()
    df['_month_idx'] = (
        (df[date_col].dt.year - min_date.year) * 12 +
        (df[date_col].dt.month - min_date.month) + 1
    )

    # Get feature columns (exclude target, date, and temporary columns)
    exclude_cols = [target_col, date_col, '_month_idx']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Create splits
    splits = {}

    for split_name, months in [
        ('train', train_months),
        ('val', val_months),
        ('test_early', test_months_early),
        ('test_late', test_months_late),
    ]:
        mask = df['_month_idx'].isin(months)
        split_df = df[mask]

        if len(split_df) == 0:
            logger.warning(f"No data for {split_name} split (months {months})")
            continue

        X = split_df[feature_cols].values
        y = split_df[target_col].values

        splits[split_name] = (X, y)
        logger.info(
            f"{split_name}: {len(split_df):,} samples, "
            f"bad rate: {y.mean():.2%}"
        )

    # Store feature names for later use
    splits['feature_names'] = feature_cols

    return splits


def random_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    exclude_cols: Optional[List[str]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create random train/validation/test splits.

    Args:
        df: DataFrame with features and target
        target_col: Target column name
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        random_state: Random seed
        exclude_cols: Additional columns to exclude from features

    Returns:
        Dictionary with 'train', 'val', 'test' tuples of (X, y)
    """
    from sklearn.model_selection import train_test_split

    # Get feature columns
    exclude = [target_col] + (exclude_cols or [])
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: separate validation from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size / (1 - test_size),
        random_state=random_state,
        stratify=y_temp
    )

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'feature_names': feature_cols,
    }
