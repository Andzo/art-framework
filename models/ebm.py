"""
Explainable Boosting Machine (EBM) Model.

Glass-box model combining GAM interpretability with gradient boosting power.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from config import RANDOM_SEED, CONFIG
from .base import BaseModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class EBMModel(BaseModel):
    """
    Explainable Boosting Machine (Glass-box Model).

    EBM is a generalized additive model (GAM) that uses gradient boosting
    to train each shape function. Key properties:

    - Inherently interpretable: f(x) = Σ fᵢ(xᵢ) + Σ fᵢⱼ(xᵢ, xⱼ)
    - Shape functions show exact feature-risk relationships
    - No need for post-hoc SHAP approximation
    - Competitive with black-box models on tabular data

    Ideal for regulatory environments (POPIA, NCA) requiring
    transparent decision explanations.

    Attributes:
        random_state: Random seed
        n_trials: Number of Optuna trials
        model: Fitted EBM model
        best_params: Best hyperparameters found
    """

    name = "EBM"

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_trials: int = None,
    ):
        """
        Initialize EBM model.

        Args:
            random_state: Random seed
            n_trials: Number of Optuna trials
        """
        self.random_state = random_state
        self.n_trials = n_trials or CONFIG.get('tuning_trials_ebm', 15)
        self.model: Optional[ExplainableBoostingClassifier] = None
        self.best_params: Optional[dict] = None

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Optuna objective function."""
        params = {
            'max_bins': trial.suggest_categorical('max_bins', [128, 256, 512]),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'interactions': trial.suggest_int('interactions', 0, 20),
            'max_rounds': trial.suggest_int('max_rounds', 2000, 15000),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'outer_bags': trial.suggest_categorical('outer_bags', [4, 8, 16]),
        }

        model = ExplainableBoostingClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=1,  # Use single thread to avoid Windows joblib memory issues
        )
        model.fit(X_train, y_train)

        # Evaluate on validation set
        from sklearn.metrics import roc_auc_score
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)

        return auc

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        tune_hyperparameters: bool = True,
    ) -> None:
        """
        Train EBM model with optional hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to perform tuning
        """
        logger.info(f"Training {self.name}")

        if tune_hyperparameters and X_val is not None and y_val is not None:
            logger.info(f"Running {self.n_trials} optimization trials")

            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(
                    trial, X_train, y_train, X_val, y_val
                ),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            self.best_params = study.best_params
            logger.info(f"Best AUC: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
        else:
            # Default parameters (dissertation optimal)
            self.best_params = {
                'max_bins': 256,
                'learning_rate': 0.01,
                'interactions': 10,
                'max_rounds': 5000,
                'min_samples_leaf': 2,
                'outer_bags': 8,
            }
            logger.info("Using default hyperparameters")

        # Train final model
        self.model = ExplainableBoostingClassifier(
            **self.best_params,
            random_state=self.random_state,
            n_jobs=1,  # Use single thread to avoid Windows joblib memory issues
        )
        self.model.fit(X_train, y_train)

        logger.info(f"{self.name} training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        if self.model is None:
            raise ValueError("Model not trained.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return feature importances from EBM.

        Args:
            feature_names: Optional feature names (uses EBM's names if None)

        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        # Get global explanation
        ebm_global = self.model.explain_global()

        names = ebm_global.data()['names']
        scores = ebm_global.data()['scores']

        importance_df = pd.DataFrame({
            'feature': names,
            'importance': scores,
        })

        return importance_df.sort_values(
            'importance', ascending=False
        ).reset_index(drop=True)

    def get_shape_functions(self) -> dict:
        """
        Extract EBM shape functions for visualization.

        Returns:
            Dictionary with feature names as keys and
            (x_values, y_values) tuples for plotting
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        ebm_global = self.model.explain_global()
        data = ebm_global.data()

        shape_functions = {}
        for i, name in enumerate(data['names']):
            shape_functions[name] = {
                'x': data['specific'][i]['names'],
                'y': data['specific'][i]['scores'],
                'density': data['specific'][i].get('density', None),
            }

        return shape_functions

    def explain_local(self, X: np.ndarray) -> List[dict]:
        """
        Get local explanations for individual predictions.

        Args:
            X: Feature matrix (n_samples x n_features)

        Returns:
            List of dictionaries with local explanations
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        explanations = []
        for i in range(len(X)):
            local_exp = self.model.explain_local(X[i:i+1])
            data = local_exp.data()

            explanations.append({
                'names': data['names'],
                'scores': data['scores'],
                'perf': data.get('perf', None),
            })

        return explanations
