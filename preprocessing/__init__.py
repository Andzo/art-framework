"""
Preprocessing module for A-R-T Framework.

Contains data loading, feature engineering, resampling, and feature reduction.
"""

from .data_loader import load_raw_data, convert_column_types, temporal_split
from .feature_engineering import (
    create_binary_flags,
    parse_activation_date,
    rename_columns,
    preprocess_data
)
from .resampling import SMOTEENNResampler, apply_smoteenn_to_splits
from .feature_reduction import FeatureReductionWorkflow

__all__ = [
    'load_raw_data',
    'convert_column_types',
    'temporal_split',
    'create_binary_flags',
    'parse_activation_date',
    'rename_columns',
    'preprocess_data',
    'SMOTEENNResampler',
    'apply_smoteenn_to_splits',
    'FeatureReductionWorkflow',
]
