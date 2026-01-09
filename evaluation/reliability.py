"""
Reliability Evaluator.

Calibration and temporal stability metrics for the Reliability pillar.
Includes ECE, Brier score, PSI, and temporal degradation analysis.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ReliabilityEvaluator:
    """
    Evaluates calibration and temporal stability.

    Provides metrics for the Reliability pillar:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Brier Score
    - Population Stability Index (PSI)
    - Temporal degradation analysis
    """

    @staticmethod
    def calculate_calibration_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict:
        """
        Calculate calibration metrics.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with ECE, MCE, Brier score, and calibration data
        """
        ece = ReliabilityEvaluator._calculate_ece(y_true, y_pred_proba, n_bins)
        mce = ReliabilityEvaluator._calculate_mce(y_true, y_pred_proba, n_bins)
        brier = np.mean((y_pred_proba - y_true) ** 2)

        # Get calibration curve data
        fraction_positives, mean_predicted = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )

        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier,
            'fraction_positives': fraction_positives,
            'mean_predicted': mean_predicted,
        }

    @staticmethod
    def _calculate_ece(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Expected Calibration Error.

        ECE measures the weighted average absolute difference between
        predicted probability and actual frequency.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            ECE value (0 = perfect calibration)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if i == n_bins - 1:  # Include upper bound for last bin
                mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

            bin_size = mask.sum()
            if bin_size > 0:
                avg_predicted = y_pred_proba[mask].mean()
                avg_actual = y_true[mask].mean()
                ece += (bin_size / n_samples) * abs(avg_actual - avg_predicted)

        return ece

    @staticmethod
    def _calculate_mce(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Maximum Calibration Error.

        MCE is the maximum absolute difference between predicted
        probability and actual frequency across all bins.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of bins

        Returns:
            MCE value (0 = perfect calibration)
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mce = 0.0

        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

            bin_size = mask.sum()
            if bin_size > 0:
                avg_predicted = y_pred_proba[mask].mean()
                avg_actual = y_true[mask].mean()
                mce = max(mce, abs(avg_actual - avg_predicted))

        return mce

    @staticmethod
    def calculate_psi(
        baseline_scores: np.ndarray,
        current_scores: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Calculate Population Stability Index.

        PSI measures distribution shift between baseline (training)
        and current (monitoring) score distributions.

        Interpretation:
        - PSI < 0.10: No significant shift
        - 0.10 ≤ PSI < 0.25: Moderate shift, investigation recommended
        - PSI ≥ 0.25: Significant shift, action required

        Args:
            baseline_scores: Scores from baseline period
            current_scores: Scores from current period
            n_bins: Number of bins

        Returns:
            PSI value
        """
        # Create bins from baseline
        bin_edges = np.percentile(baseline_scores, np.linspace(0, 100, n_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate proportions
        baseline_counts = np.histogram(baseline_scores, bins=bin_edges)[0]
        current_counts = np.histogram(current_scores, bins=bin_edges)[0]

        baseline_props = baseline_counts / len(baseline_scores)
        current_props = current_counts / len(current_scores)

        # Avoid division by zero
        baseline_props = np.clip(baseline_props, 1e-10, 1)
        current_props = np.clip(current_props, 1e-10, 1)

        # Calculate PSI
        psi = np.sum(
            (current_props - baseline_props) *
            np.log(current_props / baseline_props)
        )

        return psi

    @staticmethod
    def apply_isotonic_calibration(
        y_cal: np.ndarray,
        proba_cal: np.ndarray,
        proba_test: np.ndarray,
    ) -> np.ndarray:
        """
        Apply isotonic regression calibration.

        Args:
            y_cal: True labels from calibration set
            proba_cal: Predicted probabilities from calibration set
            proba_test: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(proba_cal, y_cal)
        return ir.transform(proba_test)

    @staticmethod
    def apply_platt_scaling(
        y_cal: np.ndarray,
        proba_cal: np.ndarray,
        proba_test: np.ndarray,
    ) -> np.ndarray:
        """
        Apply Platt scaling (logistic calibration).

        Args:
            y_cal: True labels from calibration set
            proba_cal: Predicted probabilities from calibration set
            proba_test: Probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        lr = LogisticRegression()
        lr.fit(proba_cal.reshape(-1, 1), y_cal)
        return lr.predict_proba(proba_test.reshape(-1, 1))[:, 1]

    @staticmethod
    def apply_temperature_scaling(
        logits_cal: np.ndarray,
        y_cal: np.ndarray,
        logits_test: np.ndarray,
        max_iter: int = 50,
    ) -> Tuple[np.ndarray, float]:
        """
        Apply temperature scaling for deep learning models.

        Args:
            logits_cal: Logits from calibration set
            y_cal: True labels from calibration set
            logits_test: Logits to calibrate
            max_iter: Maximum optimization iterations

        Returns:
            Tuple of (calibrated probabilities, optimal temperature)
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Convert to tensors
        logits_t = torch.FloatTensor(logits_cal)
        labels_t = torch.FloatTensor(y_cal)

        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits_t / temperature
            loss = criterion(scaled_logits, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Apply to test logits
        temp_value = temperature.item()
        calibrated_proba = 1 / (1 + np.exp(-logits_test / temp_value))

        return calibrated_proba, temp_value

    @staticmethod
    def evaluate_temporal_stability(
        model,
        data_early: Tuple[np.ndarray, np.ndarray],
        data_late: Tuple[np.ndarray, np.ndarray],
    ) -> Dict:
        """
        Evaluate model stability between early and late OOT periods.

        Args:
            model: Trained model with predict_proba method
            data_early: Tuple of (X, y) for early OOT period
            data_late: Tuple of (X, y) for late OOT period

        Returns:
            Dictionary with stability metrics
        """
        from sklearn.metrics import roc_auc_score

        X_early, y_early = data_early
        X_late, y_late = data_late

        proba_early = model.predict_proba(X_early)
        proba_late = model.predict_proba(X_late)

        auc_early = roc_auc_score(y_early, proba_early)
        auc_late = roc_auc_score(y_late, proba_late)
        auc_degradation = auc_early - auc_late

        psi = ReliabilityEvaluator.calculate_psi(proba_early, proba_late)

        ece_early = ReliabilityEvaluator._calculate_ece(y_early, proba_early)
        ece_late = ReliabilityEvaluator._calculate_ece(y_late, proba_late)

        return {
            'auc_early': auc_early,
            'auc_late': auc_late,
            'auc_degradation': auc_degradation,
            'psi': psi,
            'ece_early': ece_early,
            'ece_late': ece_late,
            'ece_degradation': ece_late - ece_early,
        }
