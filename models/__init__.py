"""
Models module for A-R-T Framework.

Contains implementations of all four model architectures:
- Logistic Regression (Elastic Net with Optuna tuning)
- XGBoost (with GPU support)
- Explainable Boosting Machine (EBM)
- FT-Transformer (Feature Tokenizer + Transformer)
"""

from .base import BaseModel
from .logistic_regression import LogisticRegressionModel
from .xgboost_model import XGBoostModel
from .ebm import EBMModel
from .ft_transformer import FTTransformerModel

__all__ = [
    'BaseModel',
    'LogisticRegressionModel',
    'XGBoostModel',
    'EBMModel',
    'FTTransformerModel',
]
