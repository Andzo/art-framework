"""
Trust Evaluator.

Dual-lens evaluation for the Trust pillar: regulatory compliance rubric
and technical explainability metrics.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class TrustEvaluator:
    """
    Evaluates explainability and regulatory compliance using dual-lens approach.

    DUAL-LENS TRUST EVALUATION FRAMEWORK
    ====================================

    Primary Assessment: Regulatory Compliance Rubric (0-9 scale)
    - Plain-language translatability (0-3)
    - Explanation stability (0-3)
    - Feature actionability (0-3)

    Secondary Assessment: Technical Explainability Metrics
    - Stability (Local Lipschitz Estimate)
    - Faithfulness (Infidelity Score)
    - Complexity (Effective Feature Count)

    Model categories:
    - Glass-box (LR, EBM): Inherently interpretable, no post-hoc required
    - Black-box (XGBoost, FT-Transformer): Require SHAP approximation
    """

    # Rubric scores by model type
    RUBRIC_SCORES = {
        'Logistic Regression': {
            'plain_language': 3,  # Coefficients directly map to odds ratios
            'stability': 3,       # Deterministic explanations
            'actionability': 3,   # Clear direction of effect
            'total': 9,
            'category': 'glass-box',
        },
        'EBM': {
            'plain_language': 3,  # Shape functions show exact relationships
            'stability': 3,       # Native explanations, no approximation
            'actionability': 3,   # Visual + tabular interpretation
            'total': 9,
            'category': 'glass-box',
        },
        'XGBoost': {
            'plain_language': 2,  # SHAP values largely acceptable with interpretation
            'stability': 2,       # SHAP can vary with background samples
            'actionability': 2,   # Direction clear but magnitude approximate
            'total': 6,
            'category': 'black-box',
        },
        'FT-Transformer': {
            'plain_language': 1,  # SHAP approximation needed
            'stability': 1,       # Higher variance in explanations
            'actionability': 1,   # Complex feature interactions
            'total': 3,
            'category': 'black-box',
        },
    }

    @staticmethod
    def regulatory_compliance_score(
        model_name: str,
        explanation_stability_score: Optional[float] = None,
    ) -> Dict:
        """
        Calculate regulatory compliance rubric score.

        Evaluates compliance with NCA Section 62 and POPIA Section 71
        through expert assessment of three dimensions.

        Args:
            model_name: Name of the model
            explanation_stability_score: Optional measured stability (0-1)

        Returns:
            Dictionary with rubric scores and compliance assessment
        """
        # Get base rubric scores
        if model_name in TrustEvaluator.RUBRIC_SCORES:
            scores = TrustEvaluator.RUBRIC_SCORES[model_name].copy()
        else:
            # Default conservative scores for unknown models
            scores = {
                'plain_language': 1,
                'stability': 1,
                'actionability': 1,
                'total': 3,
                'category': 'black-box',
            }

        # Adjust stability score if measured
        if explanation_stability_score is not None:
            if explanation_stability_score >= 0.95:
                scores['stability'] = 3
            elif explanation_stability_score >= 0.85:
                scores['stability'] = 2
            else:
                scores['stability'] = 1
            scores['total'] = (
                scores['plain_language'] +
                scores['stability'] +
                scores['actionability']
            )

        # Add compliance flags
        scores['nca_compliant'] = scores['plain_language'] >= 2
        scores['popia_compliant'] = scores['total'] >= 6
        scores['full_compliance'] = scores['nca_compliant'] and scores['popia_compliant']

        return scores

    @staticmethod
    def explain_with_shap(
        model,
        X: np.ndarray,
        feature_names: List[str],
        sample_size: int = 100,
    ) -> Dict:
        """
        Generate SHAP explanations for black-box models.

        Args:
            model: Trained model with predict_proba method
            X: Feature matrix for explanations
            feature_names: List of feature names
            sample_size: Number of samples to explain

        Returns:
            Dictionary with SHAP values and feature importance
        """
        import shap

        # Sample if needed
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        # Create explainer
        try:
            # Try TreeExplainer for tree models
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Binary classification
        except Exception:
            # Fall back to KernelExplainer
            def predict_fn(x):
                return model.predict_proba(x)

            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_sample)

        # Calculate global importance
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap,
        }).sort_values('importance', ascending=False)

        return {
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_importance': importance_df,
            'explainer': explainer,
        }

    @staticmethod
    def explain_ebm_native(
        model,
        feature_names: List[str],
    ) -> Dict:
        """
        Extract native EBM explanations using shape functions.

        Args:
            model: Trained EBM model (wrapper or raw)
            feature_names: List of feature names

        Returns:
            Dictionary with global explanation and feature importance
        """
        # Get underlying EBM model
        ebm_model = getattr(model, 'model', model)

        # Get global explanation
        global_exp = ebm_model.explain_global()
        data = global_exp.data()

        importance_df = pd.DataFrame({
            'feature': data['names'],
            'importance': data['scores'],
        }).sort_values('importance', ascending=False)

        # Extract shape functions (handle different interpret library versions)
        shape_functions = {}
        for i, name in enumerate(data['names']):
            try:
                if 'specific' in data and i < len(data['specific']):
                    specific = data['specific'][i]
                    shape_functions[name] = {
                        'x': specific.get('names', []),
                        'y': specific.get('scores', []),
                    }
                else:
                    # Fallback: try to get from explain_global with index
                    shape_functions[name] = {'x': [], 'y': []}
            except (KeyError, IndexError, TypeError):
                shape_functions[name] = {'x': [], 'y': []}

        return {
            'feature_importance': importance_df,
            'shape_functions': shape_functions,
            'global_explanation': global_exp,
        }

    @staticmethod
    def evaluate_explanation_stability(
        model,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str],
        n_bootstrap: int = 10,
    ) -> Dict:
        """
        Evaluate stability of explanations using bootstrap resampling.

        Args:
            model: Trained model
            model_name: Name of model for method selection
            X: Feature matrix
            feature_names: Feature names
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with stability metrics
        """
        # Glass-box models have perfect stability
        if model_name in ['Logistic Regression', 'EBM']:
            return {
                'stability_score': 1.0,
                'method': 'deterministic',
                'detail': 'Glass-box model with deterministic explanations',
            }

        # For black-box models, measure SHAP consistency
        all_importances = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(X), min(100, len(X)), replace=True)
            X_boot = X[indices]

            try:
                result = TrustEvaluator.explain_with_shap(
                    model, X_boot, feature_names, sample_size=50
                )
                importance = result['feature_importance']['importance'].values
                all_importances.append(importance)
            except Exception:
                continue

        if len(all_importances) < 2:
            return {
                'stability_score': np.nan,
                'method': 'shap_bootstrap',
                'detail': 'Insufficient bootstrap samples',
            }

        # Calculate pairwise rank correlations
        correlations = []
        for i in range(len(all_importances)):
            for j in range(i + 1, len(all_importances)):
                corr, _ = stats.spearmanr(all_importances[i], all_importances[j])
                if not np.isnan(corr):
                    correlations.append(corr)

        stability = np.mean(correlations) if correlations else 0.0

        return {
            'stability_score': stability,
            'method': 'shap_bootstrap',
            'n_comparisons': len(correlations),
            'detail': f'Mean Spearman correlation across {n_bootstrap} samples',
        }

    @staticmethod
    def calculate_complexity_metric(
        model,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str],
        variance_threshold: float = 0.9,
    ) -> Dict:
        """
        Calculate explanation complexity using effective feature count.

        Measures cognitive load by counting features needed to explain
        a given percentage of variance.

        Args:
            model: Trained model
            model_name: Model name
            X: Feature matrix
            feature_names: Feature names
            variance_threshold: Threshold for cumulative importance

        Returns:
            Dictionary with complexity metrics
        """
        # Get feature importance
        if model_name == 'Logistic Regression':
            coef = np.abs(model.model.coef_[0])
            importance = coef / coef.sum()
        elif model_name == 'EBM':
            result = TrustEvaluator.explain_ebm_native(model, feature_names)
            importance = result['feature_importance']['importance'].values
            importance = importance / importance.sum()
        else:
            result = TrustEvaluator.explain_with_shap(
                model, X, feature_names, sample_size=100
            )
            importance = result['feature_importance']['importance'].values
            importance = importance / importance.sum()

        # Sort and calculate cumulative
        sorted_importance = np.sort(importance)[::-1]
        cumulative = np.cumsum(sorted_importance)

        # Find number of features for threshold
        n_features_90 = np.searchsorted(cumulative, variance_threshold) + 1
        n_total = len(feature_names)

        return {
            'n_features_90pct': n_features_90,
            'n_total_features': n_total,
            'complexity_ratio': n_features_90 / n_total,
            'interpretation': 'low' if n_features_90 <= 5 else (
                'medium' if n_features_90 <= 10 else 'high'
            ),
        }

    @staticmethod
    def comprehensive_trust_assessment(
        model,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Dict:
        """
        Perform comprehensive trust evaluation.

        Combines regulatory rubric with technical metrics.

        Args:
            model: Trained model
            model_name: Model name
            X: Feature matrix
            feature_names: Feature names

        Returns:
            Dictionary with complete trust assessment
        """
        # Primary: Regulatory rubric
        stability_result = TrustEvaluator.evaluate_explanation_stability(
            model, model_name, X, feature_names
        )
        rubric = TrustEvaluator.regulatory_compliance_score(
            model_name,
            explanation_stability_score=stability_result.get('stability_score')
        )

        # Secondary: Technical metrics
        complexity = TrustEvaluator.calculate_complexity_metric(
            model, model_name, X, feature_names
        )

        return {
            'rubric': rubric,
            'stability': stability_result,
            'complexity': complexity,
            'trust_score': rubric['total'],
            'normalized_trust': rubric['total'] / 9.0,
            'model_category': rubric['category'],
        }
