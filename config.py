"""
A-R-T Framework Configuration.

Global constants and configuration for the Accuracy-Reliability-Trust
evaluation framework for credit risk models.

Author: Lebohang Andile Skungwini
"""

import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# =============================================================================
# RANDOM SEED (Global reproducibility)
# =============================================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('art_framework')


# =============================================================================
# GPU DETECTION
# =============================================================================

def check_xgboost_gpu_support() -> bool:
    """Check if XGBoost GPU support is available."""
    try:
        import xgboost as xgb
        # Try to create a small GPU-accelerated model
        params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
        data = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        xgb.train(params, data, num_boost_round=1)
        return True
    except Exception:
        return False


XGBOOST_GPU_AVAILABLE = check_xgboost_gpu_support()
TORCH_GPU_AVAILABLE = torch.cuda.is_available()

if XGBOOST_GPU_AVAILABLE:
    logger.info("XGBoost GPU support detected - will use 'gpu_hist' tree method")
else:
    logger.info("XGBoost GPU support not available - will use 'hist' tree method")

if TORCH_GPU_AVAILABLE:
    logger.info(f"PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
else:
    logger.info("PyTorch running on CPU")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory (relative to where script is run)
BASE_DIR = Path('.')

# Output directories
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = BASE_DIR / 'figures'
MODELS_DIR = BASE_DIR / 'models_saved'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'

# Create directories
for dir_path in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, CHECKPOINTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN CONFIGURATION DICTIONARY
# =============================================================================

CONFIG: Dict[str, Any] = {
    # Reproducibility
    'random_seed': RANDOM_SEED,

    # Data splits
    'test_size': 0.2,
    'val_size': 0.2,

    # Temporal split configuration (months 1-24 observation window)
    'train_months': list(range(1, 19)),      # Months 1-18
    'val_months': list(range(19, 22)),       # Months 19-21
    'test_months_early': [22, 23],           # Early OOT
    'test_months_late': [24],                # Late OOT

    # Calibration
    'calibration_bins': 10,
    'n_bootstrap_samples': 10,

    # Directories
    'results_dir': str(RESULTS_DIR),
    'figures_dir': str(FIGURES_DIR),
    'models_dir': str(MODELS_DIR),
    'checkpoints_dir': str(CHECKPOINTS_DIR),

    # Processing flags
    'use_smote_enn': True,
    'use_model_specific_preprocessing': True,
    'use_hyperparameter_tuning': True,
    'load_pretrained_models': False,

    # Hyperparameter tuning trials
    'tuning_trials_lr': 50,
    'tuning_trials_xgb': 50,
    'tuning_trials_ebm': 15,
    'tuning_trials_ftt': 20,

    # GPU settings
    'xgboost_gpu_available': XGBOOST_GPU_AVAILABLE,
    'torch_gpu_available': TORCH_GPU_AVAILABLE,
}


# =============================================================================
# COLUMN CONFIGURATION
# =============================================================================

# Default column names (can be overridden)
COLUMN_CONFIG = {
    'target': 'bad',
    'date': 'activation_date',
    'credit_score': 'credit_score',
    'score_band': 'score_band',
}

# Columns to drop during preprocessing
COLUMNS_TO_DROP = [
    'netpay_to_inv_l6m',
    'avg30plus_l3m',
    'times2_l12m',
]

# Column name mappings (from raw to clean)
COLUMN_RENAME_MAP = {
    'ClientType': 'client_type',
    'contracttype': 'contract_type',
    'financed_deal': 'financed_deal_flag',
    'conn_date': 'activation_date',
    'Act_Port': 'portfolio',
    'VAP_SabreScore_TM': 'credit_score',
    'Risk_Band': 'score_band',
    'monthssinceaccountcreation': 'months_since_acc_creation',
    'rdinmonth': 'returned_debit_flag',
    'ARG': 'internal_risk_ranking',
    'obs_numpayreversals': 'total_payment_reversals',
    'obs_amountdueinclvat': 'invoice_amount',
    'GMIPValue': 'monthly_income',
    'GMIPBand': 'income_band',
    'obs_noactivesubscribers': 'no_active_subs',
}


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAMES = [
    'Logistic Regression',
    'XGBoost',
    'EBM',
    'FT-Transformer',
]

MODEL_COLORS = {
    'Logistic Regression': '#1f77b4',  # Blue
    'XGBoost': '#ff7f0e',              # Orange
    'EBM': '#2ca02c',                  # Green
    'FT-Transformer': '#d62728',       # Red
}

MODEL_SHORT_NAMES = {
    'Logistic Regression': 'LR',
    'XGBoost': 'XGB',
    'EBM': 'EBM',
    'FT-Transformer': 'FTT',
}


# =============================================================================
# FEATURE REDUCTION THRESHOLDS (for Logistic Regression)
# =============================================================================

FEATURE_REDUCTION_CONFIG = {
    'pearson_threshold': 0.7,      # Numerical correlation threshold
    'cramers_v_threshold': 0.7,    # Categorical association threshold
    'eta_threshold': 0.7,          # Mixed-type correlation threshold
    'vif_threshold': 10.0,         # VIF multicollinearity threshold
    'min_diff': 0.02,              # Minimum target correlation difference
    'high_cardinality_threshold': 10,  # For encoding strategy
}
