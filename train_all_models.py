"""
Dissertation-Aligned Credit Risk Model Training Script.

Trains all 4 models (LR, XGBoost, EBM, FT-Transformer) with EXACT alignment
to dissertation methodology. Supports optional hyperparameter tuning.

Usage:
    python train_all_models.py --data-dir ../art_source_data --output-dir ./results
    python train_all_models.py --data-dir ../art_source_data --enable-tuning true

Author: Lebohang Andile Skungwini
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG, RANDOM_SEED

# Set seeds early
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_presplit_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load pre-split CSV files (train_df.csv, val_df.csv, test_df.csv).
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    data_path = Path(data_dir)
    
    logger.info(f"Loading data from {data_path}")
    
    train_df = pd.read_csv(data_path / 'train_df.csv')
    val_df = pd.read_csv(data_path / 'val_df.csv')
    test_df = pd.read_csv(data_path / 'test_df.csv')
    
    logger.info(f"Train: {len(train_df):,} rows, Val: {len(val_df):,} rows, Test: {len(test_df):,} rows")
    logger.info(f"Train bad rate: {train_df['bad'].mean():.2%}")
    logger.info(f"Val bad rate: {val_df['bad'].mean():.2%}")
    logger.info(f"Test bad rate: {test_df['bad'].mean():.2%}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
    }


def prepare_features(
    df: pd.DataFrame,
    target_col: str = 'bad',
    exclude_cols: List[str] = None,
    scaler_params: Dict = None,
    fit_scaler: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
    """
    Prepare features for model training.
    
    Handles:
    - Dropping target and excluded columns
    - Encoding categoricals (one-hot for low cardinality, frequency for high)
    - Missing value imputation (median for numeric, mode for categorical)
    - Z-score scaling
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        scaler_params: Pre-computed mean/std for scaling (from training set)
        fit_scaler: Whether to fit scaler parameters
        
    Returns:
        Tuple of (X, y, feature_names, scaler_params)
    """
    df = df.copy()
    exclude_cols = exclude_cols or ['activation_date']
    
    # Extract target
    y = df[target_col].values
    
    # Drop target and excluded columns
    drop_cols = [target_col] + [c for c in exclude_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    # Identify column types
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Handle missing values in numeric columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Handle missing values in categorical columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # Encode categoricals
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            # Frequency encoding
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0)
            df = df.rename(columns={col: f"{col}_freq"})
    
    # Get feature names before scaling
    feature_names = df.columns.tolist()
    
    # Z-score scaling
    if fit_scaler:
        scaler_params = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
        }
    
    if scaler_params:
        for col in df.columns:
            if col in scaler_params['mean']:
                mean = scaler_params['mean'][col]
                std = scaler_params['std'].get(col, 1.0)
                if std > 0:
                    df[col] = (df[col] - mean) / std
    
    X = df.values.astype(np.float32)
    
    # Handle any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, feature_names, scaler_params


def apply_smote_enn(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE-ENN resampling to training data.
    
    SMOTE: k_neighbors=5
    ENN: n_neighbors=3
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Resampled (X, y)
    """
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    
    logger.info(f"Before SMOTE-ENN: {len(y):,} samples, bad rate: {y.mean():.2%}")
    
    smote_enn = SMOTEENN(
        smote=SMOTE(k_neighbors=5, random_state=RANDOM_SEED),
        enn=EditedNearestNeighbours(n_neighbors=3, sampling_strategy='all'),
        random_state=RANDOM_SEED,
    )
    
    X_resampled, y_resampled = smote_enn.fit_resample(X, y)
    
    logger.info(f"After SMOTE-ENN: {len(y_resampled):,} samples, bad rate: {y_resampled.mean():.2%}")
    
    return X_resampled, y_resampled


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enable_tuning: bool = True,
) -> object:
    """Train Elastic Net Logistic Regression."""
    from models.logistic_regression import LogisticRegressionModel
    
    logger.info("=" * 60)
    logger.info("Training Logistic Regression")
    logger.info("=" * 60)
    
    model = LogisticRegressionModel(
        random_state=RANDOM_SEED,
        n_trials=CONFIG.get('tuning_trials_lr', 50),
    )
    model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=enable_tuning)
    
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enable_tuning: bool = True,
) -> object:
    """Train XGBoost model."""
    from models.xgboost_model import XGBoostModel
    
    logger.info("=" * 60)
    logger.info("Training XGBoost")
    logger.info("=" * 60)
    
    model = XGBoostModel(
        random_state=RANDOM_SEED,
        n_trials=CONFIG.get('tuning_trials_xgb', 50),
    )
    model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=enable_tuning)
    
    return model


def train_ebm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enable_tuning: bool = True,
) -> object:
    """Train Explainable Boosting Machine."""
    from models.ebm import EBMModel
    
    logger.info("=" * 60)
    logger.info("Training EBM")
    logger.info("=" * 60)
    
    model = EBMModel(
        random_state=RANDOM_SEED,
        n_trials=CONFIG.get('tuning_trials_ebm', 15),
    )
    model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=enable_tuning)
    
    return model


def train_ft_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    enable_tuning: bool = True,
) -> object:
    """Train FT-Transformer."""
    from models.ft_transformer import FTTransformerModel
    
    logger.info("=" * 60)
    logger.info("Training FT-Transformer")
    logger.info("=" * 60)
    
    model = FTTransformerModel(
        n_features=X_train.shape[1],
        random_state=RANDOM_SEED,
        n_trials=CONFIG.get('tuning_trials_ftt', 20),
    )
    model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=enable_tuning)
    
    return model


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model,
    model_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
) -> Dict:
    """
    Evaluate a model using the full A-R-T framework.
    
    Returns dictionary with all metrics.
    """
    from evaluation.accuracy import AccuracyEvaluator
    from evaluation.reliability import ReliabilityEvaluator
    from evaluation.trust import TrustEvaluator
    
    logger.info(f"Evaluating {model_name}")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test)
    
    # Accuracy pillar
    accuracy_metrics = AccuracyEvaluator.calculate_metrics(y_test, y_pred_proba, y_pred)
    
    # Reliability pillar
    reliability_metrics = ReliabilityEvaluator.calculate_calibration_metrics(y_test, y_pred_proba)
    
    # Trust pillar
    trust_metrics = TrustEvaluator.comprehensive_trust_assessment(
        model, model_name, X_test[:1000], feature_names
    )
    
    results = {
        'model': model_name,
        # Accuracy
        'auc': accuracy_metrics['auc'],
        'gini': accuracy_metrics['gini'],
        'ks': accuracy_metrics['ks_statistic'],
        'top_decile_capture': accuracy_metrics['top_decile_capture'],
        'precision': accuracy_metrics['precision'],
        'recall': accuracy_metrics['recall'],
        'f1': accuracy_metrics['f1'],
        # Reliability
        'ece': reliability_metrics['ece'],
        'mce': reliability_metrics['mce'],
        'brier': reliability_metrics['brier_score'],
        # Trust
        'trust_score': trust_metrics['trust_score'],
        'trust_normalized': trust_metrics['normalized_trust'],
        'model_category': trust_metrics['model_category'],
        # Raw predictions for further analysis
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
    }
    
    logger.info(f"  AUC: {results['auc']:.4f}, Gini: {results['gini']:.4f}, KS: {results['ks']:.4f}")
    logger.info(f"  ECE: {results['ece']:.4f}, Brier: {results['brier']:.4f}")
    logger.info(f"  Trust Score: {results['trust_score']}/9 ({results['model_category']})")
    
    return results


def run_delong_tests(results: Dict, y_test: np.ndarray) -> pd.DataFrame:
    """Run DeLong tests for all model pairs."""
    from evaluation.accuracy import AccuracyEvaluator
    
    model_names = list(results.keys())
    comparisons = []
    
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            
            test_result = AccuracyEvaluator.delong_test(
                y_test,
                results[m1]['y_pred_proba'],
                results[m2]['y_pred_proba'],
            )
            
            comparisons.append({
                'model_1': m1,
                'model_2': m2,
                'auc_1': test_result['auc_1'],
                'auc_2': test_result['auc_2'],
                'auc_diff': test_result['auc_diff'],
                'z_statistic': test_result['z_statistic'],
                'p_value': test_result['p_value'],
                'significant': test_result['p_value'] < 0.05 if not np.isnan(test_result['p_value']) else False,
            })
    
    return pd.DataFrame(comparisons)


# =============================================================================
# VISUALIZATION
# =============================================================================

def generate_visualizations(
    models: Dict,
    results: Dict,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
):
    """Generate all visualizations."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix
    from sklearn.calibration import calibration_curve
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    logger.info("Generating visualizations...")
    
    # Color scheme
    colors = {
        'Logistic Regression': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'EBM': '#2ca02c',
        'FT-Transformer': '#d62728',
    }
    
    # 1. Combined ROC curves
    plt.figure(figsize=(10, 8))
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={res['auc']:.4f})", 
                 color=colors.get(model_name, 'gray'), linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - All Models', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / 'all_models_roc.png', dpi=150)
    plt.close()
    
    # 2. Individual model plots
    for model_name, res in results.items():
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        
        # ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
        plt.plot(fpr, tpr, color=colors.get(model_name, 'blue'), linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve (AUC={res["auc"]:.4f})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f'{safe_name}_roc_curve.png', dpi=150)
        plt.close()
        
        # Calibration curve
        plt.figure(figsize=(8, 6))
        fraction_pos, mean_pred = calibration_curve(y_test, res['y_pred_proba'], n_bins=10)
        plt.plot(mean_pred, fraction_pos, 's-', label=model_name, color=colors.get(model_name, 'blue'))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'{model_name} Calibration Curve (ECE={res["ece"]:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / f'{safe_name}_calibration.png', dpi=150)
        plt.close()
        
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, res['y_pred'])
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} Confusion Matrix')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', fontsize=14)
        plt.xticks([0, 1], ['Non-Default', 'Default'])
        plt.yticks([0, 1], ['Non-Default', 'Default'])
        plt.tight_layout()
        plt.savefig(figures_dir / f'{safe_name}_confusion_matrix.png', dpi=150)
        plt.close()
    
    # 3. A-R-T Radar chart
    try:
        from math import pi
        
        categories = ['Accuracy\n(AUC)', 'Reliability\n(1-ECE)', 'Trust\n(Score/9)']
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the polygon
        
        for model_name, res in results.items():
            values = [
                res['auc'],
                1 - res['ece'],
                res['trust_normalized'],
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
                    color=colors.get(model_name, 'gray'))
            ax.fill(angles, values, alpha=0.1, color=colors.get(model_name, 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.title('A-R-T Framework Comparison', fontsize=14, y=1.08)
        plt.tight_layout()
        plt.savefig(figures_dir / 'art_radar_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Could not generate radar chart: {e}")
    
    # 4. EBM shape functions (if EBM model exists)
    if 'EBM' in models:
        try:
            ebm_model = models['EBM']
            shape_funcs = ebm_model.get_shape_functions()
            
            # Plot top 6 shape functions
            top_features = list(shape_funcs.keys())[:6]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for idx, feat_name in enumerate(top_features):
                ax = axes[idx // 3, idx % 3]
                data = shape_funcs[feat_name]
                ax.plot(data['x'], data['y'], linewidth=2)
                ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel(feat_name, fontsize=10)
                ax.set_ylabel('Log-odds contribution', fontsize=10)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle('EBM Shape Functions (Top 6 Features)', fontsize=14)
            plt.tight_layout()
            plt.savefig(figures_dir / 'ebm_shape_functions.png', dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate EBM shape functions: {e}")
    
    # 5. LR coefficients
    if 'Logistic Regression' in models:
        try:
            lr_model = models['Logistic Regression']
            importance_df = lr_model.get_feature_importance(feature_names)
            top_15 = importance_df.head(15)
            
            plt.figure(figsize=(12, 8))
            colors_coef = ['#d62728' if c > 0 else '#1f77b4' for c in top_15['coefficient']]
            plt.barh(range(len(top_15)), top_15['coefficient'], color=colors_coef)
            plt.yticks(range(len(top_15)), top_15['feature'])
            plt.xlabel('Coefficient')
            plt.title('Logistic Regression - Top 15 Feature Coefficients')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(figures_dir / 'lr_coefficients.png', dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not generate LR coefficients plot: {e}")
    
    # 6. SHAP summary for XGBoost
    if 'XGBoost' in models:
        try:
            import shap
            # Suppress verbose SHAP logging
            logging.getLogger('shap').setLevel(logging.WARNING)
            logger.info("Generating XGBoost SHAP summary...")
            
            xgb_model = models['XGBoost']
            # Sample for SHAP (use subset for speed)
            X_sample = X_test[:500] if len(X_test) > 500 else X_test
            
            explainer = shap.TreeExplainer(xgb_model.model)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                             show=False, max_display=15)
            plt.title('XGBoost SHAP Summary', fontsize=14)
            plt.tight_layout()
            plt.savefig(figures_dir / 'xgboost_shap_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("XGBoost SHAP summary saved")
        except Exception as e:
            logger.warning(f"Could not generate XGBoost SHAP plot: {e}")
    
    # 7. SHAP summary for FT-Transformer
    if 'FT-Transformer' in models:
        try:
            import shap
            logger.info("Generating FT-Transformer SHAP summary...")
            
            ftt_model = models['FT-Transformer']
            # Sample for SHAP (smaller for neural network)
            X_sample = X_test[:200] if len(X_test) > 200 else X_test
            background = X_sample[:50]
            
            def ftt_predict(x):
                return ftt_model.predict_proba(x)
            
            explainer = shap.KernelExplainer(ftt_predict, background)
            shap_values = explainer.shap_values(X_sample[:100], nsamples=100)
            
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_sample[:100], feature_names=feature_names,
                             show=False, max_display=15)
            plt.title('FT-Transformer SHAP Summary', fontsize=14)
            plt.tight_layout()
            plt.savefig(figures_dir / 'fttransformer_shap_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            logger.info("FT-Transformer SHAP summary saved")
        except Exception as e:
            logger.warning(f"Could not generate FT-Transformer SHAP plot: {e}")
    
    # 8. FT-Transformer Attention Heatmap
    if 'FT-Transformer' in models:
        try:
            logger.info("Generating FT-Transformer attention heatmap...")
            
            ftt_model = models['FT-Transformer']
            # Get a sample to visualize attention
            X_sample = X_test[:10]
            
            # Hook to capture attention weights
            attention_weights = []
            def attention_hook(module, input, output):
                # Capture attention weights
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    attention_weights.append(output.detach().cpu().numpy())
            
            # Register hooks on transformer layers
            ftt_model.model.eval()
            hooks = []
            for layer in ftt_model.model.transformer.layers:
                hook = layer.self_attn.register_forward_hook(attention_hook)
                hooks.append(hook)
            
            # Forward pass
            X_tensor = torch.FloatTensor(X_sample).to(ftt_model.device)
            with torch.no_grad():
                _ = ftt_model.model(X_tensor)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Plot average attention across samples and layers
            if attention_weights:
                avg_attention = np.mean([np.mean(aw, axis=0) for aw in attention_weights], axis=0)
                
                # Select subset of features for readability
                n_display = min(20, len(feature_names))
                feature_labels = ['[CLS]'] + feature_names[:n_display-1]
                
                plt.figure(figsize=(14, 12))
                plt.imshow(avg_attention[:n_display, :n_display], cmap='Blues', aspect='auto')
                plt.colorbar(label='Attention Weight')
                plt.xticks(range(n_display), feature_labels, rotation=45, ha='right', fontsize=8)
                plt.yticks(range(n_display), feature_labels, fontsize=8)
                plt.xlabel('Key Features', fontsize=12)
                plt.ylabel('Query Features', fontsize=12)
                plt.title('FT-Transformer Attention Heatmap (Averaged)', fontsize=14)
                plt.tight_layout()
                plt.savefig(figures_dir / 'ftt_attention_heatmap.png', dpi=150, bbox_inches='tight')
                plt.close()
                logger.info("FT-Transformer attention heatmap saved")
        except Exception as e:
            logger.warning(f"Could not generate attention heatmap: {e}")
    
    # 9. Comparative Calibration Plot (all models)
    try:
        logger.info("Generating comparative calibration plot...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, res) in enumerate(results.items()):
            ax = axes[idx]
            fraction_pos, mean_pred = calibration_curve(y_test, res['y_pred_proba'], n_bins=10)
            
            ax.plot(mean_pred, fraction_pos, 's-', color=colors.get(model_name, 'blue'), 
                   linewidth=2, markersize=8, label=model_name)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7)
            ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.2, 
                           color=colors.get(model_name, 'blue'))
            ax.set_xlabel('Mean Predicted Probability', fontsize=11)
            ax.set_ylabel('Fraction of Positives', fontsize=11)
            ax.set_title(f'{model_name}\nECE = {res["ece"]:.4f}', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        plt.suptitle('Calibration Comparison Across Models', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(figures_dir / 'comparative_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("Comparative calibration plot saved")
    except Exception as e:
        logger.warning(f"Could not generate comparative calibration plot: {e}")
    
    logger.info(f"Visualizations saved to {figures_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def save_results(
    models: Dict,
    results: Dict,
    delong_df: pd.DataFrame,
    output_dir: Path,
):
    """Save all results and models."""
    # Create directories
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    for model_name, model in models.items():
        safe_name = model_name.lower().replace(' ', '_').replace('-', '_')
        
        if model_name == 'FT-Transformer':
            torch.save(model.model.state_dict(), models_dir / f'{safe_name}.pt')
            # Also save model config
            config = {
                'n_features': model.n_features,
                'best_params': model.best_params,
            }
            with open(models_dir / f'{safe_name}_config.json', 'w') as f:
                json.dump(config, f, indent=2)
        else:
            with open(models_dir / f'{safe_name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Saved {model_name} model")
    
    # Save metrics CSV
    metrics_rows = []
    for model_name, res in results.items():
        metrics_rows.append({
            'Model': model_name,
            'AUC': res['auc'],
            'Gini': res['gini'],
            'KS': res['ks'],
            'Top_Decile_Capture': res['top_decile_capture'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1'],
            'ECE': res['ece'],
            'MCE': res['mce'],
            'Brier': res['brier'],
            'Trust_Score': res['trust_score'],
            'Trust_Normalized': res['trust_normalized'],
            'Model_Category': res['model_category'],
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / 'art_metrics.csv', index=False)
    logger.info(f"Saved metrics to {output_dir / 'art_metrics.csv'}")
    
    # Save DeLong tests
    delong_df.to_csv(output_dir / 'delong_tests.csv', index=False)
    logger.info(f"Saved DeLong tests to {output_dir / 'delong_tests.csv'}")
    
    # Generate summary report
    report = [
        "=" * 60,
        "A-R-T EVALUATION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "ACCURACY PILLAR (Discrimination)",
        "-" * 40,
    ]
    
    for model_name, res in results.items():
        report.append(f"{model_name}:")
        report.append(f"  AUC: {res['auc']:.4f}, Gini: {res['gini']:.4f}, KS: {res['ks']:.4f}")
    
    report.extend([
        "",
        "RELIABILITY PILLAR (Calibration)",
        "-" * 40,
    ])
    
    for model_name, res in results.items():
        report.append(f"{model_name}:")
        report.append(f"  ECE: {res['ece']:.4f}, Brier: {res['brier']:.4f}")
    
    report.extend([
        "",
        "TRUST PILLAR (Explainability)",
        "-" * 40,
    ])
    
    for model_name, res in results.items():
        report.append(f"{model_name}:")
        report.append(f"  Score: {res['trust_score']}/9, Category: {res['model_category']}")
    
    report.extend([
        "",
        "DELONG TEST RESULTS",
        "-" * 40,
    ])
    
    for _, row in delong_df.iterrows():
        sig = "**" if row['significant'] else ""
        report.append(f"{row['model_1']} vs {row['model_2']}: "
                     f"ΔAU= {row['auc_diff']:.4f}, p={row['p_value']:.4f} {sig}")
    
    report.append("")
    report.append("=" * 60)
    
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print summary
    print('\n'.join(report))


def main():
    parser = argparse.ArgumentParser(
        description='Train all credit risk models with dissertation methodology'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../art_source_data',
        help='Directory containing train_df.csv, val_df.csv, test_df.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory for outputs (models, metrics, figures)'
    )
    parser.add_argument(
        '--enable-tuning',
        type=str,
        default='false',
        choices=['true', 'false'],
        help='Enable hyperparameter tuning (true/false)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of models to train (lr,xgb,ebm,ftt) or "all"'
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    enable_tuning = args.enable_tuning.lower() == 'true'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("=" * 60)
    logger.info("A-R-T FRAMEWORK CREDIT RISK MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Hyperparameter tuning: {'ENABLED' if enable_tuning else 'DISABLED'}")
    logger.info(f"Random seed: {args.seed}")
    
    # Parse model selection
    if args.models.lower() == 'all':
        train_models = ['lr', 'xgb', 'ebm', 'ftt']
    else:
        train_models = [m.strip().lower() for m in args.models.split(',')]
    
    logger.info(f"Models to train: {train_models}")
    
    # Step 1: Load data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Loading pre-split data")
    logger.info("=" * 60)
    
    data = load_presplit_data(args.data_dir)
    
    # Step 2: Prepare features
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Preparing features")
    logger.info("=" * 60)
    
    X_train, y_train, feature_names, scaler_params = prepare_features(
        data['train'], fit_scaler=True
    )
    X_val, y_val, _, _ = prepare_features(
        data['val'], scaler_params=scaler_params
    )
    X_test, y_test, _, _ = prepare_features(
        data['test'], scaler_params=scaler_params
    )
    
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Train shape: {X_train.shape}")
    logger.info(f"Val shape: {X_val.shape}")
    logger.info(f"Test shape: {X_test.shape}")
    
    # Step 3: Apply SMOTE-ENN to training data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Applying SMOTE-ENN to training data")
    logger.info("=" * 60)
    
    X_train_resampled, y_train_resampled = apply_smote_enn(X_train, y_train)
    
    # Step 4: Train models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Training models")
    logger.info("=" * 60)
    
    models = {}
    
    if 'lr' in train_models:
        models['Logistic Regression'] = train_logistic_regression(
            X_train_resampled, y_train_resampled, X_val, y_val, enable_tuning
        )
    
    if 'xgb' in train_models:
        models['XGBoost'] = train_xgboost(
            X_train_resampled, y_train_resampled, X_val, y_val, enable_tuning
        )
    
    if 'ebm' in train_models:
        models['EBM'] = train_ebm(
            X_train_resampled, y_train_resampled, X_val, y_val, enable_tuning
        )
    
    if 'ftt' in train_models:
        models['FT-Transformer'] = train_ft_transformer(
            X_train_resampled, y_train_resampled, X_val, y_val, enable_tuning
        )
    
    # Step 5: Evaluate models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Evaluating models on test set")
    logger.info("=" * 60)
    
    results = {}
    for model_name, model in models.items():
        results[model_name] = evaluate_model(model, model_name, X_test, y_test, feature_names)
    
    # Step 6: DeLong tests
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Running DeLong tests")
    logger.info("=" * 60)
    
    delong_df = run_delong_tests(results, y_test)
    
    # Step 7: Generate visualizations
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Generating visualizations")
    logger.info("=" * 60)
    
    generate_visualizations(models, results, y_test, feature_names, output_dir)
    
    # Step 8: Save results
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: Saving results")
    logger.info("=" * 60)
    
    save_results(models, results, delong_df, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
