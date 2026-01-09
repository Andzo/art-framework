"""
Comparative Plots Module.

Visualizations for Chapter 6 cross-model comparisons:
radar charts, Pareto frontiers, temporal degradation.

Author: Lebohang Andile Skungwini
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, MODEL_COLORS, MODEL_SHORT_NAMES

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300


def plot_radar_chart(
    pillar_scores: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot A-R-T radar/spider chart comparing all models.

    Args:
        pillar_scores: DataFrame with columns ['Model', 'Accuracy', 'Reliability', 'Trust']
                      Values should be normalized to 0-1 range
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'art_radar_chart.png')

    categories = ['Accuracy', 'Reliability', 'Trust']
    n_cats = len(categories)

    # Compute angles
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for _, row in pillar_scores.iterrows():
        model_name = row['Model']
        values = [row['Accuracy'], row['Reliability'], row['Trust']]
        values += values[:1]  # Close the polygon

        color = MODEL_COLORS.get(model_name, None)
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)

        ax.plot(angles, values, 'o-', linewidth=2, color=color, label=short_name)
        ax.fill(angles, values, alpha=0.25, color=color)

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.title('A-R-T Pillar Comparison', y=1.08, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved radar chart to {save_path}")
    return Path(save_path)


def plot_pareto_frontier(
    metrics: Dict[str, Dict],
    x_metric: str = 'auc',
    y_metric: str = 'trust_score',
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot Pareto frontier showing accuracy vs trust trade-off.

    Args:
        metrics: Dictionary with model metrics
        x_metric: Metric for x-axis (default: 'auc')
        y_metric: Metric for y-axis (default: 'trust_score')
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'pareto_frontier.png')

    fig, ax = plt.subplots(figsize=(10, 8))

    x_values = []
    y_values = []
    names = []

    for model_name, m in metrics.items():
        x = m.get(x_metric, 0)
        y = m.get(y_metric, 0)

        color = MODEL_COLORS.get(model_name, '#333333')
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)

        ax.scatter(x, y, s=200, c=color, edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(
            short_name, (x, y),
            xytext=(10, 10), textcoords='offset points',
            fontsize=11, fontweight='bold'
        )

        x_values.append(x)
        y_values.append(y)
        names.append(model_name)

    # Draw Pareto frontier
    pareto_points = []
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        dominated = False
        for j, (x2, y2) in enumerate(zip(x_values, y_values)):
            if i != j and x2 >= x and y2 >= y and (x2 > x or y2 > y):
                dominated = True
                break
        if not dominated:
            pareto_points.append((x, y))

    if len(pareto_points) > 1:
        pareto_points.sort(key=lambda p: p[0])
        pareto_x = [p[0] for p in pareto_points]
        pareto_y = [p[1] for p in pareto_points]
        ax.plot(pareto_x, pareto_y, 'g--', linewidth=2, alpha=0.7, label='Pareto Frontier')

    ax.set_xlabel(f'{x_metric.upper().replace("_", " ")}', fontsize=12)
    ax.set_ylabel(f'{y_metric.upper().replace("_", " ")}', fontsize=12)
    ax.set_title('Accuracy vs Trust Trade-off', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved Pareto frontier to {save_path}")
    return Path(save_path)


def plot_temporal_degradation(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> Path:
    """
    Plot temporal performance degradation across periods.

    Args:
        results: Dictionary with 'val_auc', 'early_oot_auc', 'late_oot_auc' per model
        save_path: Path to save figure

    Returns:
        Path to saved figure
    """
    save_path = save_path or str(FIGURES_DIR / 'temporal_degradation.png')

    periods = ['Validation', 'Early OOT', 'Late OOT']
    x = np.arange(len(periods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (model_name, res) in enumerate(results.items()):
        aucs = [
            res.get('val_auc', 0),
            res.get('early_oot_auc', 0),
            res.get('late_oot_auc', 0),
        ]

        color = MODEL_COLORS.get(model_name, None)
        short_name = MODEL_SHORT_NAMES.get(model_name, model_name)

        offset = (i - len(results) / 2 + 0.5) * width
        bars = ax.bar(x + offset, aucs, width, label=short_name, color=color, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, aucs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=8
            )

    ax.set_xlabel('Time Period')
    ax.set_ylabel('AUC')
    ax.set_title('Temporal Performance Degradation')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend(loc='lower left')
    ax.set_ylim([0.5, 1.0])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved temporal degradation plot to {save_path}")
    return Path(save_path)


def generate_ranking_table(
    metrics: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate ranking shift table comparing AUC-only vs A-R-T composite rankings.

    Args:
        metrics: Dictionary with model metrics including 'auc', 'ece', 'trust_score'
        save_path: Optional path to save as CSV

    Returns:
        DataFrame with ranking comparison
    """
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            'Model': model_name,
            'AUC': m.get('auc', 0),
            'ECE': m.get('ece', 0),
            'Trust': m.get('trust_score', 0),
        })

    df = pd.DataFrame(rows)

    # Compute ranks
    df['AUC_Rank'] = df['AUC'].rank(ascending=False).astype(int)
    df['Composite_Rank'] = (
        df['AUC'].rank(ascending=False) +
        df['ECE'].rank(ascending=True) +
        df['Trust'].rank(ascending=False)
    ).rank().astype(int)

    df['Rank_Shift'] = df['AUC_Rank'] - df['Composite_Rank']

    # Sort by composite rank
    df = df.sort_values('Composite_Rank')

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Saved ranking table to {save_path}")

    return df
