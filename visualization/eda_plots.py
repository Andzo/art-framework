"""
EDA Plots Module.

Visualizations for Chapter 2 exploratory data analysis.

Author: Lebohang Andile Skungwini
"""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from config import FIGURES_DIR

logger = logging.getLogger(__name__)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def plot_monthly_activations(
    df: pd.DataFrame,
    date_col: str,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot monthly account activations over time.

    Args:
        df: DataFrame with activation dates
        date_col: Date column name
        save_path: Path to save figure (default: figures/monthly_activations.png)

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'monthly_activations.png')

    # Aggregate by month
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.to_period('M')
    monthly = df_copy.groupby('month').size().reset_index(name='count')
    monthly['month'] = monthly['month'].dt.to_timestamp()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(monthly['month'], monthly['count'], width=20, color='#1f77b4', alpha=0.8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Activations')
    ax.set_title('Monthly Account Activations')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved monthly activations plot to {save_path}")

    return Path(save_path)


def plot_target_distribution(
    y: np.ndarray,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot distribution of target variable.

    Args:
        y: Target array
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'target_distribution.png')

    counts = np.bincount(y.astype(int))
    labels = ['Good (0)', 'Bad (1)']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=['#2ca02c', '#d62728'], alpha=0.8)

    # Add percentage labels
    total = counts.sum()
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f'{count:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )

    ax.set_ylabel('Count')
    ax.set_title('Target Distribution (DPD60+ within 9 months)')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved target distribution plot to {save_path}")

    return Path(save_path)


def plot_monthly_bad_rate(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot monthly bad rate over time.

    Args:
        df: DataFrame with dates and target
        date_col: Date column name
        target_col: Target column name
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'monthly_bad_rate.png')

    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy[date_col]).dt.to_period('M')

    monthly_rate = df_copy.groupby('month')[target_col].mean().reset_index()
    monthly_rate['month'] = monthly_rate['month'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_rate['month'], monthly_rate[target_col] * 100,
            marker='o', linewidth=2, color='#d62728')
    ax.fill_between(monthly_rate['month'], monthly_rate[target_col] * 100,
                    alpha=0.3, color='#d62728')

    ax.set_xlabel('Month')
    ax.set_ylabel('Bad Rate (%)')
    ax.set_title('Monthly Default Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved monthly bad rate plot to {save_path}")

    return Path(save_path)


def plot_correlation_matrix(
    df: pd.DataFrame,
    features: List[str],
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot correlation matrix heatmap for numerical variables.

    Args:
        df: DataFrame with features
        features: List of numerical feature names
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'correlation_matrix.png')

    # Calculate correlation
    corr_matrix = df[features].corr()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title('Pearson Correlation Matrix')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved correlation matrix to {save_path}")

    return Path(save_path)


def plot_cramers_v_heatmap(
    df: pd.DataFrame,
    categorical_features: List[str],
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot Cramér's V association matrix for categorical variables.

    Args:
        df: DataFrame with categorical features
        categorical_features: List of categorical feature names
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'cramers_v_heatmap.png')

    def cramers_v(x, y):
        """Calculate Cramér's V statistic."""
        contingency = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(contingency)[0]
        n = len(x)
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * min_dim))

    # Calculate pairwise Cramér's V
    n = len(categorical_features)
    matrix = np.zeros((n, n))

    for i, col1 in enumerate(categorical_features):
        for j, col2 in enumerate(categorical_features):
            if i <= j:
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    v = cramers_v(df[col1], df[col2])
                    matrix[i, j] = v
                    matrix[j, i] = v

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=categorical_features,
        yticklabels=categorical_features,
        square=True,
        linewidths=0.5,
        ax=ax,
    )

    ax.set_title("Cramér's V Association Matrix")
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved Cramér's V heatmap to {save_path}")

    return Path(save_path)


def plot_numerical_distributions(
    df: pd.DataFrame,
    target_col: str,
    features: List[str],
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot distributions of numerical variables by target class.

    Args:
        df: DataFrame with features and target
        target_col: Target column name
        features: List of numerical features to plot
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'numerical_distributions.png')

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        # Box plots by target
        df_plot = df[[feature, target_col]].dropna()
        df_plot[target_col] = df_plot[target_col].map({0: 'Good', 1: 'Bad'})

        sns.boxplot(
            data=df_plot,
            x=target_col,
            y=feature,
            ax=ax,
            palette={'Good': '#2ca02c', 'Bad': '#d62728'}
        )
        ax.set_title(feature)
        ax.set_xlabel('')

    # Hide empty subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('Numerical Distributions by Target', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved numerical distributions to {save_path}")

    return Path(save_path)
