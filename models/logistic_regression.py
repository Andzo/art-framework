"""
Logistic Regression Model.

Elastic Net Logistic Regression with Optuna hyperparameter optimization.
Combines L1 (Lasso) and L2 (Ridge) regularization for robustness.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import RANDOM_SEED, CONFIG
from .base import BaseModel

logger = logging.getLogger(__name__)

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class LogisticRegressionModel(BaseModel):
    """
    Elastic Net Logistic Regression with Optuna Hyperparameter Optimization.

    Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization:
    - L1 enables feature selection (sparse coefficients)
    - L2 handles multicollinearity and groups of correlated features

    The loss function with Elastic Net is:
        L = CrossEntropy + λ * [α * ||w||₁ + (1-α) * ||w||₂²]

    Where:
        - λ = 1/C (inverse regularization strength)
        - α = l1_ratio (mixing parameter)

    Attributes:
        random_state: Random seed
        n_trials: Number of Optuna trials
        model: Fitted LogisticRegression model
        best_params: Best hyperparameters found
        optimization_history: Trial history for analysis
    """

    name = "Logistic Regression"

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        n_trials: int = None,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            random_state: Random seed for reproducibility
            n_trials: Number of Optuna optimization trials
        """
        self.random_state = random_state
        self.n_trials = n_trials or CONFIG.get('tuning_trials_lr', 50)
        self.model: Optional[LogisticRegression] = None
        self.best_params: Optional[dict] = None
        self.optimization_history: List[dict] = []

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Optuna objective function for hyperparameter optimization."""
        # Define search space
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

        # Create and train model
        model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=C,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
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
        Train the model with optional Optuna hyperparameter optimization.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (required for tuning)
            y_val: Validation target (required for tuning)
            tune_hyperparameters: Whether to perform hyperparameter tuning
        """
        logger.info(f"Training {self.name}")

        if tune_hyperparameters and X_val is not None and y_val is not None:
            # Run Optuna optimization
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
            self.optimization_history = [
                {
                    'trial': t.number,
                    'value': t.value,
                    **t.params,
                }
                for t in study.trials
            ]

            logger.info(f"Best AUC: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")
        else:
            # Use default parameters
            self.best_params = {'C': 1.0, 'l1_ratio': 0.5}
            logger.info("Using default hyperparameters")

        # Train final model with best parameters
        self.model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=self.best_params['C'],
            l1_ratio=self.best_params['l1_ratio'],
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        logger.info(f"{self.name} training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
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
        Return feature importances (coefficients) sorted by absolute value.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature names, coefficients, and odds ratios
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        coef = self.model.coef_[0]

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(coef))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef),
            'odds_ratio': np.exp(coef),
        })

        return importance_df.sort_values(
            'abs_coefficient', ascending=False
        ).reset_index(drop=True)

    def get_optimization_report(self) -> pd.DataFrame:
        """Return optimization history as DataFrame."""
        return pd.DataFrame(self.optimization_history)

    def save_optimization_results(self, filepath: str) -> None:
        """Save optimization results to CSV."""
        report = self.get_optimization_report()
        if not report.empty:
            report.to_csv(filepath, index=False)
            logger.info(f"Optimization results saved to {filepath}")
