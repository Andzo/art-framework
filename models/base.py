"""
Base Model Abstract Class.

Defines the interface that all model implementations must follow.

Author: Lebohang Andile Skungwini
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.

    All models must implement:
    - train(): Fit the model to training data
    - predict_proba(): Return probability estimates
    - predict(): Return class predictions

    Attributes:
        name: Human-readable model name
        model: The underlying model object (set after training)
    """

    name: str = "BaseModel"
    model = None

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, for early stopping)
            y_val: Validation target (optional)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of positive class.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities for positive class (shape: n_samples,)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Array of class predictions (0 or 1)
        """
        pass
