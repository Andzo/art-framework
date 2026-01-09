"""
Visualization module for A-R-T Framework.

Contains plotting functions for:
- EDA plots (Chapter 2)
- Model-specific plots (Chapters 3-5)
- Comparative analysis plots (Chapter 6)
"""

from .eda_plots import (
    plot_monthly_activations,
    plot_target_distribution,
    plot_monthly_bad_rate,
    plot_correlation_matrix,
    plot_cramers_v_heatmap,
    plot_numerical_distributions,
)
from .model_plots import (
    plot_roc_curve,
    plot_all_models_roc,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_lr_coefficients,
    plot_shap_summary,
    plot_ebm_shape_functions,
)
from .comparative_plots import (
    plot_radar_chart,
    plot_pareto_frontier,
    plot_temporal_degradation,
    generate_ranking_table,
)

__all__ = [
    # EDA
    'plot_monthly_activations',
    'plot_target_distribution',
    'plot_monthly_bad_rate',
    'plot_correlation_matrix',
    'plot_cramers_v_heatmap',
    'plot_numerical_distributions',
    # Model plots
    'plot_roc_curve',
    'plot_all_models_roc',
    'plot_calibration_curve',
    'plot_confusion_matrix',
    'plot_lr_coefficients',
    'plot_shap_summary',
    'plot_ebm_shape_functions',
    # Comparative
    'plot_radar_chart',
    'plot_pareto_frontier',
    'plot_temporal_degradation',
    'generate_ranking_table',
]
