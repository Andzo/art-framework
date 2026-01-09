"""
Model Plots Module.

Visualizations for model-specific results: ROC curves, calibration,
confusion matrices, feature importance, and SHAP summaries.

Author: Lebohang Andile Skungwini
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc

from config import FIGURES_DIR, MODEL_COLORS

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot ROC curve for a single model.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_roc_curve.png"
    save_path = save_path or str(FIGURES_DIR / filename)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1

    fig, ax = plt.subplots(figsize=(8, 6))

    color = MODEL_COLORS.get(model_name, '#1f77b4')
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{model_name} (AUC={roc_auc:.4f}, Gini={gini:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3, color=color)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved ROC curve to {save_path}")
    return Path(save_path)


def plot_all_models_roc(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot ROC curves for all models on the same plot.

    Args:
        results: Dictionary with model results containing 'y_true' and 'y_pred_proba'
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'all_models_roc.png')

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, res in results.items():
        y_true = res['y_true']
        y_prob = res['y_pred_proba']

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        color = MODEL_COLORS.get(model_name, None)
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f'{model_name} (AUC={roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved all models ROC to {save_path}")
    return Path(save_path)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot calibration curve (reliability diagram) for a single model.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        model_name: Name of the model
        n_bins: Number of calibration bins
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_calibration.png"
    save_path = save_path or str(FIGURES_DIR / filename)

    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Calibration curve
    color = MODEL_COLORS.get(model_name, '#1f77b4')
    ax1.plot(mean_predicted, fraction_positive, 's-', color=color,
             label=model_name, markersize=8)
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')

    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title(f'Calibration Curve - {model_name}')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of predictions
    ax2.hist(y_prob, bins=50, alpha=0.7, color=color, edgecolor='white')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Predictions')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved calibration curve to {save_path}")
    return Path(save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot confusion matrix for a model.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_confusion_matrix.png"
    save_path = save_path or str(FIGURES_DIR / filename)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1'],
        ax=ax, cbar=True,
        annot_kws={'size': 14}
    )

    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")
    return Path(save_path)


def plot_lr_coefficients(
    model,
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot Logistic Regression coefficients.

    Args:
        model: Trained LR model with .model attribute
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'lr_coefficients.png')

    # Get coefficients
    lr_model = getattr(model, 'model', model)
    coef = lr_model.coef_[0]

    # Create DataFrame and sort
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    }).sort_values('abs_coefficient', ascending=True).tail(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#d62728' if c > 0 else '#2ca02c' for c in coef_df['coefficient']]
    ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Coefficient')
    ax.set_title(f'Top {top_n} Logistic Regression Coefficients')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved LR coefficients to {save_path}")
    return Path(save_path)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot SHAP summary plot.

    Args:
        shap_values: SHAP values array
        X: Feature matrix
        feature_names: List of feature names
        model_name: Name of model
        top_n: Number of top features
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    import shap

    filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_shap_summary.png"
    save_path = save_path or str(FIGURES_DIR / filename)

    fig, ax = plt.subplots(figsize=(10, 8))

    shap.summary_plot(
        shap_values, X,
        feature_names=feature_names,
        max_display=top_n,
        show=False
    )

    plt.title(f'SHAP Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved SHAP summary to {save_path}")
    return Path(save_path)


def plot_ebm_shape_functions(
    model,
    top_n: int = 6,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot EBM shape functions for top features.

    Args:
        model: Trained EBM model
        top_n: Number of top features to plot
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'ebm_shape_functions.png')

    ebm_model = getattr(model, 'model', model)
    global_exp = ebm_model.explain_global()
    data = global_exp.data()

    # Get top features by importance
    importance = list(zip(data['names'], data['scores']))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = [name for name, _ in importance[:top_n]]

    # Create subplots
    n_cols = min(3, len(top_features))
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, feature in enumerate(top_features):
        idx = data['names'].index(feature)
        specific = data['specific'][idx]

        x_vals = specific['names']
        y_vals = specific['scores']

        ax = axes[i]
        ax.plot(range(len(x_vals)), y_vals, color='#2ca02c', linewidth=2)
        ax.fill_between(range(len(x_vals)), y_vals, alpha=0.3, color='#2ca02c')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        ax.set_title(feature)
        ax.set_xlabel('Feature Value')
        ax.set_ylabel('Contribution to Log-Odds')

    for i in range(len(top_features), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('EBM Shape Functions', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved EBM shape functions to {save_path}")
    return Path(save_path)
