"""
XGBoost Model.

Gradient boosting model with GPU support and Optuna hyperparameter optimization.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from config import RANDOM_SEED, CONFIG, XGBOOST_GPU_AVAILABLE
from .base import BaseModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostModel(BaseModel):
    """
    XGBoost Gradient Boosting Model.

    Uses GPU acceleration when available (gpu_hist tree method).
    Includes Optuna hyperparameter optimization.

    Attributes:
        random_state: Random seed
        n_trials: Number of Optuna trials
        use_gpu: Whether to use GPU acceleration
        model: Fitted XGBoost Booster
        best_params: Best hyperparameters found
        best_iteration: Best iteration from early stopping
    """

    name = "XGBoost"

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_trials: int = None,
        use_gpu: bool = None,
    ):
        """
        Initialize XGBoost model.

        Args:
            random_state: Random seed
            n_trials: Number of Optuna trials
            use_gpu: Force GPU usage (None = auto-detect)
        """
        self.random_state = random_state
        self.n_trials = n_trials or CONFIG.get('tuning_trials_xgb', 50)
        self.use_gpu = use_gpu if use_gpu is not None else XGBOOST_GPU_AVAILABLE
        self.model: Optional[xgb.Booster] = None
        self.best_params: Optional[dict] = None
        self.best_iteration: Optional[int] = None

    def _get_base_params(self) -> dict:
        """Get base parameters including GPU settings."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'seed': self.random_state,
            'verbosity': 0,
        }

        if self.use_gpu:
            params['tree_method'] = 'gpu_hist'
            params['device'] = 'cuda'
        else:
            params['tree_method'] = 'hist'

        return params

    def _objective(
        self,
        trial: optuna.Trial,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
    ) -> float:
        """Optuna objective function."""
        params = self._get_base_params()
        params.update({
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        })

        # Handle class imbalance
        scale_pos_weight = (
            (dtrain.get_label() == 0).sum() /
            (dtrain.get_label() == 1).sum()
        )
        params['scale_pos_weight'] = scale_pos_weight

        # Train with early stopping
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        # Get validation AUC
        val_auc = bst.best_score
        return val_auc

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune_hyperparameters: bool = True,
    ) -> None:
        """
        Train XGBoost model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to perform tuning
        """
        logger.info(f"Training {self.name}" +
                    (" with GPU" if self.use_gpu else " on CPU"))

        dtrain = xgb.DMatrix(X_train, label=y_train)

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
        else:
            dval = dtrain

        if tune_hyperparameters and X_val is not None:
            logger.info(f"Running {self.n_trials} optimization trials")

            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, dtrain, dval),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            self.best_params = study.best_params
            logger.info(f"Best AUC: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
        else:
            # Default parameters
            self.best_params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 1e-5,
                'reg_lambda': 1.0,
            }
            logger.info("Using default hyperparameters")

        # Train final model
        params = self._get_base_params()
        params.update(self.best_params)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = scale_pos_weight

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )

        self.best_iteration = self.model.best_iteration
        logger.info(
            f"{self.name} training complete "
            f"(best iteration: {self.best_iteration})"
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        if self.model is None:
            raise ValueError("Model not trained.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        importance_type: str = 'gain',
    ) -> pd.DataFrame:
        """
        Return feature importances.

        Args:
            feature_names: List of feature names
            importance_type: 'gain', 'weight', or 'cover'

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        importance = self.model.get_score(importance_type=importance_type)

        if feature_names is not None:
            # Map f0, f1, etc. to actual names
            importance = {
                feature_names[int(k[1:])]: v
                for k, v in importance.items()
                if k.startswith('f') and k[1:].isdigit()
            }

        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])

        return importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)
