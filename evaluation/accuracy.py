"""
Accuracy Evaluator.

Discrimination metrics for the Accuracy pillar of A-R-T framework.
Includes AUC, Gini, KS statistic, and DeLong test for model comparison.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class AccuracyEvaluator:
    """
    Evaluates discrimination performance metrics.

    Provides comprehensive metrics for the Accuracy pillar:
    - AUC-ROC and Gini coefficient
    - Kolmogorov-Smirnov statistic
    - Precision, Recall, F1 score
    - Top-decile capture rate
    - DeLong test for AUC comparison
    """

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Calculate comprehensive accuracy metrics.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            y_pred: Predicted class labels (computed if None)
            threshold: Classification threshold

        Returns:
            Dictionary with all accuracy metrics
        """
        if y_pred is None:
            y_pred = (y_pred_proba >= threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc - 1
        ks = AccuracyEvaluator._calculate_ks_statistic(y_true, y_pred_proba)
        top_decile = AccuracyEvaluator._top_decile_capture(y_true, y_pred_proba)

        metrics = {
            'auc': auc,
            'gini': gini,
            'ks_statistic': ks,
            'top_decile_capture': top_decile,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'average_precision': average_precision_score(y_true, y_pred_proba),
        }

        return metrics

    @staticmethod
    def _calculate_ks_statistic(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> float:
        """
        Calculate Kolmogorov-Smirnov statistic.

        KS measures the maximum separation between cumulative distributions
        of positive and negative class scores.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            KS statistic (0 to 1)
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        return np.max(tpr - fpr)

    @staticmethod
    def _top_decile_capture(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> float:
        """
        Calculate percentage of positives captured in top decile.

        This metric shows what fraction of actual positives (defaults)
        are found in the highest 10% of predicted scores.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            Capture rate (0 to 1)
        """
        n_samples = len(y_true)
        top_decile_size = int(n_samples * 0.1)

        # Get indices of top 10% predicted scores
        top_indices = np.argsort(y_pred_proba)[-top_decile_size:]

        # Count positives in top decile
        positives_in_top = y_true[top_indices].sum()
        total_positives = y_true.sum()

        if total_positives == 0:
            return 0.0

        return positives_in_top / total_positives

    @staticmethod
    def delong_test(
        y_true: np.ndarray,
        y_pred_proba_1: np.ndarray,
        y_pred_proba_2: np.ndarray,
    ) -> Dict:
        """
        DeLong's test for comparing two AUC-ROC values.

        Tests the null hypothesis that two ROC curves have equal AUC.
        Uses the DeLong et al. (1988) variance estimation.

        Args:
            y_true: True binary labels
            y_pred_proba_1: Predictions from model 1
            y_pred_proba_2: Predictions from model 2

        Returns:
            Dictionary with test statistic, p-value, and AUC difference
        """
        # Separate positive and negative samples
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_scores_1 = y_pred_proba_1[pos_mask]
        neg_scores_1 = y_pred_proba_1[neg_mask]
        pos_scores_2 = y_pred_proba_2[pos_mask]
        neg_scores_2 = y_pred_proba_2[neg_mask]

        n_pos = len(pos_scores_1)
        n_neg = len(neg_scores_1)

        if n_pos == 0 or n_neg == 0:
            return {'z_statistic': np.nan, 'p_value': np.nan, 'auc_diff': np.nan}

        # Calculate AUCs using Mann-Whitney
        auc_1 = roc_auc_score(y_true, y_pred_proba_1)
        auc_2 = roc_auc_score(y_true, y_pred_proba_2)
        auc_diff = auc_1 - auc_2

        # Compute structural components for variance
        def compute_structural_components(pos_scores, neg_scores):
            """Compute V10 and V01 components."""
            V10 = np.zeros(n_pos)
            V01 = np.zeros(n_neg)

            for i in range(n_pos):
                V10[i] = np.mean(pos_scores[i] > neg_scores) + \
                         0.5 * np.mean(pos_scores[i] == neg_scores)

            for j in range(n_neg):
                V01[j] = np.mean(pos_scores > neg_scores[j]) + \
                         0.5 * np.mean(pos_scores == neg_scores[j])

            return V10, V01

        V10_1, V01_1 = compute_structural_components(pos_scores_1, neg_scores_1)
        V10_2, V01_2 = compute_structural_components(pos_scores_2, neg_scores_2)

        # Covariance matrix of (AUC1, AUC2)
        S10 = np.cov(np.vstack([V10_1, V10_2]))
        S01 = np.cov(np.vstack([V01_1, V01_2]))

        # Variance of difference
        S = S10 / n_pos + S01 / n_neg
        var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

        if var_diff <= 0:
            return {'z_statistic': np.nan, 'p_value': np.nan, 'auc_diff': auc_diff}

        # Z-statistic
        z = auc_diff / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return {
            'z_statistic': z,
            'p_value': p_value,
            'auc_diff': auc_diff,
            'auc_1': auc_1,
            'auc_2': auc_2,
        }

    @staticmethod
    def compare_models_statistically(
        models_results: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Compare all model pairs using DeLong test.

        Args:
            models_results: Dictionary with model results containing
                           'y_true' and 'y_pred_proba' for each model

        Returns:
            DataFrame with pairwise DeLong comparisons
        """
        model_names = list(models_results.keys())
        comparisons = []

        # Get common y_true (should be same for all models)
        y_true = None
        for results in models_results.values():
            if 'y_true' in results:
                y_true = results['y_true']
                break

        if y_true is None:
            raise ValueError("No y_true found in model results")

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model_1 = model_names[i]
                model_2 = model_names[j]

                proba_1 = models_results[model_1]['y_pred_proba']
                proba_2 = models_results[model_2]['y_pred_proba']

                result = AccuracyEvaluator.delong_test(y_true, proba_1, proba_2)

                comparisons.append({
                    'model_1': model_1,
                    'model_2': model_2,
                    'auc_1': result['auc_1'],
                    'auc_2': result['auc_2'],
                    'auc_diff': result['auc_diff'],
                    'z_statistic': result['z_statistic'],
                    'p_value': result['p_value'],
                    'significant': result['p_value'] < 0.05 if result['p_value'] is not np.nan else False,
                })

        return pd.DataFrame(comparisons)

    @staticmethod
    def get_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Get confusion matrix as dictionary.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with TN, FP, FN, TP
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}
