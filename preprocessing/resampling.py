"""
SMOTE-ENN Resampling Module.

Implements SMOTE-ENN hybrid resampling for handling class imbalance
in credit risk datasets.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

from config import RANDOM_SEED

logger = logging.getLogger(__name__)


class SMOTEENNResampler:
    """
    SMOTE-ENN Hybrid Resampling for Class Imbalance.

    Combines two complementary techniques:
    1. SMOTE (Synthetic Minority Over-sampling Technique): Generates synthetic
       minority class samples using k-nearest neighbors interpolation
    2. ENN (Edited Nearest Neighbours): Cleans noisy samples by removing
       instances whose class differs from the majority of their neighbors

    This hybrid approach addresses class imbalance while reducing overfitting
    to noisy synthetic samples.

    Attributes:
        random_state: Random seed for reproducibility
        sampling_strategy: SMOTE sampling strategy
        k_neighbors_smote: Number of neighbors for SMOTE
        n_neighbors_enn: Number of neighbors for ENN
        resampling_stats: Dictionary with resampling statistics
    """

    def __init__(
        self,
        random_state: int = RANDOM_SEED,
        sampling_strategy: str = 'auto',
        k_neighbors_smote: int = 5,
        n_neighbors_enn: int = 3,
    ):
        """
        Initialize SMOTE-ENN resampler.

        Args:
            random_state: Random seed for reproducibility
            sampling_strategy: 'auto' = resample minority to match majority
            k_neighbors_smote: Number of neighbors for SMOTE (default: 5)
            n_neighbors_enn: Number of neighbors for ENN cleaning (default: 3)
        """
        self.random_state = random_state
        self.sampling_strategy = sampling_strategy
        self.k_neighbors_smote = k_neighbors_smote
        self.n_neighbors_enn = n_neighbors_enn

        # Initialize resampler
        self.resampler = SMOTEENN(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors_smote,
                random_state=random_state,
            ),
            enn=EditedNearestNeighbours(
                n_neighbors=n_neighbors_enn,
                sampling_strategy='all',
            ),
            random_state=random_state,
        )

        self.resampling_stats: Optional[Dict] = None

    def fit_resample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE-ENN resampling to training data.

        Args:
            X: Feature matrix (training data only)
            y: Target vector (training data only)

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        # Record original statistics
        original_counts = np.bincount(y.astype(int))
        original_ratio = original_counts[1] / original_counts[0]

        logger.info(
            f"Original: {len(y):,} samples, "
            f"Class 0: {original_counts[0]:,}, "
            f"Class 1: {original_counts[1]:,}, "
            f"Ratio: {original_ratio:.4f}"
        )

        # Apply SMOTE-ENN
        X_resampled, y_resampled = self.resampler.fit_resample(X, y)

        # Record resampled statistics
        resampled_counts = np.bincount(y_resampled.astype(int))
        resampled_ratio = resampled_counts[1] / resampled_counts[0]

        logger.info(
            f"Resampled: {len(y_resampled):,} samples, "
            f"Class 0: {resampled_counts[0]:,}, "
            f"Class 1: {resampled_counts[1]:,}, "
            f"Ratio: {resampled_ratio:.4f}"
        )

        # Store statistics
        self.resampling_stats = {
            'original_samples': len(y),
            'original_class_0': int(original_counts[0]),
            'original_class_1': int(original_counts[1]),
            'original_ratio': float(original_ratio),
            'resampled_samples': len(y_resampled),
            'resampled_class_0': int(resampled_counts[0]),
            'resampled_class_1': int(resampled_counts[1]),
            'resampled_ratio': float(resampled_ratio),
            'samples_added': len(y_resampled) - len(y),
            'synthetic_minority': int(resampled_counts[1] - original_counts[1]),
        }

        return X_resampled, y_resampled

    def get_report(self) -> Optional[pd.DataFrame]:
        """
        Return resampling statistics as DataFrame.

        Returns:
            DataFrame with resampling statistics or None if not yet fit
        """
        if self.resampling_stats is None:
            return None

        return pd.DataFrame([self.resampling_stats])

    def save_report(self, filepath: str) -> None:
        """
        Save resampling report to CSV.

        Args:
            filepath: Path to save CSV file
        """
        report = self.get_report()
        if report is not None:
            report.to_csv(filepath, index=False)
            logger.info(f"Resampling report saved to {filepath}")


def apply_smoteenn_to_splits(
    data_splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    random_state: int = RANDOM_SEED,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict]:
    """
    Apply SMOTE-ENN to training data only, preserving validation/test distributions.

    This is the recommended way to use SMOTE-ENN in the training pipeline:
    - Training data is resampled to balance classes
    - Validation and test data are left unchanged to reflect true distribution

    Args:
        data_splits: Dictionary with 'train', 'val', 'test' keys
                     Each value is a tuple of (X, y)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of:
        - Updated data_splits with resampled training data
        - Resampling report dictionary
    """
    if 'train' not in data_splits:
        raise ValueError("data_splits must contain 'train' key")

    X_train, y_train = data_splits['train']

    # Apply SMOTE-ENN
    resampler = SMOTEENNResampler(random_state=random_state)
    X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)

    # Update splits
    data_splits = data_splits.copy()
    data_splits['train'] = (X_train_resampled, y_train_resampled)

    return data_splits, resampler.resampling_stats
