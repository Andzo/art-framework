"""
Generate Plots from Saved Models.

Loads trained models from disk and generates all visualizations
without retraining.

Usage:
    python generate_plots.py --models-dir ./results/models --output-dir ./results

Author: Lebohang Andile Skungwini
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_saved_models(models_dir: Path) -> Dict:
    """Load all saved models from directory."""
    models = {}
    
    # Load sklearn/interpret models (pickle)
    for pkl_file in models_dir.glob('*.pkl'):
        model_name = pkl_file.stem.replace('_', ' ').title()
        # Fix naming
        if 'Logistic' in model_name:
            model_name = 'Logistic Regression'
        elif 'Xgboost' in model_name:
            model_name = 'XGBoost'
        elif 'Ebm' in model_name:
            model_name = 'EBM'
        
        with open(pkl_file, 'rb') as f:
            models[model_name] = pickle.load(f)
        logger.info(f"Loaded {model_name} from {pkl_file}")
    
    # Load FT-Transformer (PyTorch)
    pt_file = models_dir / 'ft_transformer.pt'
    config_file = models_dir / 'ft_transformer_config.json'
    
    if pt_file.exists() and config_file.exists():
        from models.ft_transformer import FTTransformerModel, FTTransformer
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        ftt_model = FTTransformerModel(
            n_features=config['n_features'],
            random_state=RANDOM_SEED,
        )
        
        # Create the underlying model with saved params
        arch_params = {k: v for k, v in config['best_params'].items() 
                      if k in ['d_model', 'n_heads', 'n_layers', 'dropout']}
        ftt_model.model = FTTransformer(
            n_features=config['n_features'],
            **arch_params
        ).to(ftt_model.device)
        
        # Load weights
        ftt_model.model.load_state_dict(torch.load(pt_file, map_location=ftt_model.device))
        ftt_model.model.eval()
        
        models['FT-Transformer'] = ftt_model
        logger.info(f"Loaded FT-Transformer from {pt_file}")
    
    return models


def prepare_test_data(data_dir: str) -> tuple:
    """Load and prepare test data."""
    from train_all_models import load_presplit_data, prepare_features
    
    data = load_presplit_data(data_dir)
    
    # Get scaler params from training data
    _, _, feature_names, scaler_params = prepare_features(
        data['train'], fit_scaler=True
    )
    
    # Prepare test data
    X_test, y_test, _, _ = prepare_features(
        data['test'], scaler_params=scaler_params
    )
    
    return X_test, y_test, feature_names


def evaluate_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> Dict:
    """Evaluate all loaded models."""
    from evaluation.accuracy import AccuracyEvaluator
    from evaluation.reliability import ReliabilityEvaluator
    from evaluation.trust import TrustEvaluator
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        accuracy_metrics = AccuracyEvaluator.calculate_metrics(y_test, y_pred_proba, y_pred)
        reliability_metrics = ReliabilityEvaluator.calculate_calibration_metrics(y_test, y_pred_proba)
        trust_metrics = TrustEvaluator.comprehensive_trust_assessment(
            model, model_name, X_test[:1000], feature_names
        )
        
        results[model_name] = {
            'auc': accuracy_metrics['auc'],
            'gini': accuracy_metrics['gini'],
            'ks': accuracy_metrics['ks_statistic'],
            'top_decile_capture': accuracy_metrics['top_decile_capture'],
            'precision': accuracy_metrics['precision'],
            'recall': accuracy_metrics['recall'],
            'f1': accuracy_metrics['f1'],
            'ece': reliability_metrics['ece'],
            'mce': reliability_metrics['mce'],
            'brier': reliability_metrics['brier_score'],
            'trust_score': trust_metrics['trust_score'],
            'trust_normalized': trust_metrics['normalized_trust'],
            'model_category': trust_metrics['model_category'],
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred,
        }
        
        logger.info(f"  AUC: {results[model_name]['auc']:.4f}, ECE: {results[model_name]['ece']:.4f}")
    
    return results


def generate_all_plots(
    models: Dict,
    results: Dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
):
    """Generate all visualizations from saved models."""
    from train_all_models import generate_visualizations
    
    # Use the same visualization function from train_all_models
    generate_visualizations(models, results, y_test, feature_names, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots from saved models (no retraining)'
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./results/models',
        help='Directory containing saved models'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../art_source_data',
        help='Directory containing test data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./results',
        help='Directory for output figures'
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("GENERATING PLOTS FROM SAVED MODELS")
    logger.info("=" * 60)
    
    # Step 1: Load models
    logger.info("\nStep 1: Loading saved models...")
    models = load_saved_models(models_dir)
    
    if not models:
        logger.error(f"No models found in {models_dir}")
        return
    
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Step 2: Load test data
    logger.info("\nStep 2: Loading test data...")
    X_test, y_test, feature_names = prepare_test_data(args.data_dir)
    logger.info(f"Test data: {X_test.shape[0]} samples, {len(feature_names)} features")
    
    # Step 3: Evaluate models
    logger.info("\nStep 3: Evaluating models...")
    results = evaluate_models(models, X_test, y_test, feature_names)
    
    # Step 4: Generate plots
    logger.info("\nStep 4: Generating visualizations...")
    generate_all_plots(models, results, X_test, y_test, feature_names, output_dir)
    
    logger.info("\n" + "=" * 60)
    logger.info("PLOT GENERATION COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Figures saved to: {output_dir / 'figures'}")


if __name__ == '__main__':
    main()
