"""
A-R-T Evaluation Pipeline.

Main orchestrator for the Accuracy-Reliability-Trust evaluation framework.
Handles data loading, preprocessing, model training, evaluation, and reporting.

Author: Lebohang Andile Skungwini
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from config import CONFIG, RANDOM_SEED, MODELS_DIR, RESULTS_DIR, CHECKPOINTS_DIR

# Import modules
from preprocessing import (
    load_raw_data,
    convert_column_types,
    preprocess_data,
    temporal_split,
    random_split,
    SMOTEENNResampler,
    FeatureReductionWorkflow,
)
from models import (
    LogisticRegressionModel,
    XGBoostModel,
    EBMModel,
    FTTransformerModel,
)
from evaluation import (
    AccuracyEvaluator,
    ReliabilityEvaluator,
    TrustEvaluator,
)
from visualization import (
    plot_all_models_roc,
    plot_roc_curve,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_lr_coefficients,
    plot_ebm_shape_functions,
    plot_radar_chart,
    plot_temporal_degradation,
    generate_ranking_table,
)

logger = logging.getLogger(__name__)


class ARTEvaluationPipeline:
    """
    Main pipeline for Accuracy-Reliability-Trust evaluation.

    Orchestrates the complete workflow:
    1. Data loading and preprocessing
    2. Model training with optional hyperparameter tuning
    3. Accuracy pillar evaluation (AUC, Gini, KS)
    4. Reliability pillar evaluation (ECE, PSI, temporal stability)
    5. Trust pillar evaluation (rubric + technical explainability)
    6. Visualization and report generation

    Features:
    - Checkpoint save/restore for long-running pipelines
    - Model persistence (pickle, torch)
    - Comprehensive logging

    Attributes:
        config: Configuration dictionary
        models: Dictionary of trained models
        results: Dictionary of evaluation results
        data: Dictionary with train/val/test splits
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary (uses default if None)
        """
        self.config = config or CONFIG
        self.models: Dict = {}
        self.results: Dict = {}
        self.data: Dict = {}
        self.feature_names: List[str] = []

        # Set random seeds
        np.random.seed(self.config.get('random_seed', RANDOM_SEED))
        torch.manual_seed(self.config.get('random_seed', RANDOM_SEED))

    def load_and_prepare_data(
        self,
        data_path: str,
        target_col: str = 'bad',
        date_col: Optional[str] = 'activation_date',
        use_temporal_split: bool = True,
    ) -> Dict:
        """
        Load data and prepare train/val/test splits.

        Args:
            data_path: Path to CSV data file
            target_col: Target column name
            date_col: Date column for temporal split
            use_temporal_split: Use temporal or random split

        Returns:
            Dictionary with data splits
        """
        logger.info(f"Loading data from {data_path}")

        # Load and preprocess
        df = load_raw_data(data_path)
        df = convert_column_types(df)
        df = preprocess_data(df, target_col, date_col)

        # Split data
        if use_temporal_split and date_col:
            splits = temporal_split(
                df, date_col, target_col,
                train_months=self.config['train_months'],
                val_months=self.config['val_months'],
                test_months_early=self.config['test_months_early'],
                test_months_late=self.config['test_months_late'],
            )
        else:
            splits = random_split(
                df, target_col,
                test_size=self.config['test_size'],
                val_size=self.config['val_size'],
            )

        self.feature_names = splits.get('feature_names', [])
        self.data = splits

        # Apply SMOTE-ENN if configured
        if self.config.get('use_smote_enn', True):
            logger.info("Applying SMOTE-ENN resampling")
            self.data, smote_stats = SMOTEENNResampler().fit_resample(
                *self.data['train']
            )

        return self.data

    def train_models(
        self,
        tune_hyperparameters: bool = True,
    ) -> Dict:
        """
        Train all four models.

        Args:
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary of trained models
        """
        X_train, y_train = self.data['train']
        X_val, y_val = self.data.get('val', (None, None))

        logger.info("Training models...")

        # Logistic Regression (with feature reduction)
        logger.info("Training Logistic Regression")
        lr_model = LogisticRegressionModel()
        lr_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.models['Logistic Regression'] = lr_model

        # XGBoost
        logger.info("Training XGBoost")
        xgb_model = XGBoostModel()
        xgb_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.models['XGBoost'] = xgb_model

        # EBM
        logger.info("Training EBM")
        ebm_model = EBMModel()
        ebm_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.models['EBM'] = ebm_model

        # FT-Transformer
        logger.info("Training FT-Transformer")
        ftt_model = FTTransformerModel(n_features=X_train.shape[1])
        ftt_model.train(X_train, y_train, X_val, y_val, tune_hyperparameters)
        self.models['FT-Transformer'] = ftt_model

        logger.info("All models trained")
        return self.models

    def evaluate_accuracy(self, split: str = 'test_early') -> Dict:
        """
        Evaluate Accuracy pillar for all models.

        Args:
            split: Which data split to evaluate on

        Returns:
            Dictionary with accuracy metrics per model
        """
        X, y = self.data.get(split, (None, None))
        if X is None:
            logger.warning(f"Split {split} not found")
            return {}

        accuracy_results = {}
        for model_name, model in self.models.items():
            y_pred_proba = model.predict_proba(X)
            y_pred = model.predict(X)

            metrics = AccuracyEvaluator.calculate_metrics(y, y_pred_proba, y_pred)
            metrics['y_true'] = y
            metrics['y_pred_proba'] = y_pred_proba
            metrics['y_pred'] = y_pred

            accuracy_results[model_name] = metrics
            logger.info(
                f"{model_name}: AUC={metrics['auc']:.4f}, "
                f"Gini={metrics['gini']:.4f}, KS={metrics['ks_statistic']:.4f}"
            )

        self.results['accuracy'] = accuracy_results
        return accuracy_results

    def evaluate_reliability(self) -> Dict:
        """
        Evaluate Reliability pillar for all models.

        Returns:
            Dictionary with reliability metrics per model
        """
        reliability_results = {}

        for model_name, model in self.models.items():
            # Calibration on validation set
            X_val, y_val = self.data.get('val', (None, None))
            if X_val is not None:
                y_pred_proba = model.predict_proba(X_val)
                cal_metrics = ReliabilityEvaluator.calculate_calibration_metrics(
                    y_val, y_pred_proba
                )

            # Temporal stability
            if 'test_early' in self.data and 'test_late' in self.data:
                stability = ReliabilityEvaluator.evaluate_temporal_stability(
                    model,
                    self.data['test_early'],
                    self.data['test_late'],
                )
            else:
                stability = {}

            reliability_results[model_name] = {
                **cal_metrics,
                **stability,
            }

            logger.info(
                f"{model_name}: ECE={cal_metrics.get('ece', 0):.4f}, "
                f"Brier={cal_metrics.get('brier_score', 0):.4f}"
            )

        self.results['reliability'] = reliability_results
        return reliability_results

    def evaluate_trust(self) -> Dict:
        """
        Evaluate Trust pillar for all models.

        Returns:
            Dictionary with trust metrics per model
        """
        X_val, _ = self.data.get('val', (None, None))
        trust_results = {}

        for model_name, model in self.models.items():
            trust = TrustEvaluator.comprehensive_trust_assessment(
                model, model_name, X_val, self.feature_names
            )
            trust_results[model_name] = trust

            logger.info(
                f"{model_name}: Trust Score={trust['trust_score']}/9, "
                f"Category={trust['model_category']}"
            )

        self.results['trust'] = trust_results
        return trust_results

    def generate_visualizations(self) -> List[Path]:
        """
        Generate all evaluation visualizations.

        Returns:
            List of paths to generated figures
        """
        figures = []

        # ROC curves
        if 'accuracy' in self.results:
            figures.append(plot_all_models_roc(self.results['accuracy']))

            for model_name, res in self.results['accuracy'].items():
                figures.append(plot_roc_curve(
                    res['y_true'], res['y_pred_proba'], model_name
                ))
                figures.append(plot_calibration_curve(
                    res['y_true'], res['y_pred_proba'], model_name
                ))
                figures.append(plot_confusion_matrix(
                    res['y_true'], res['y_pred'], model_name
                ))

        # Model-specific plots
        if 'Logistic Regression' in self.models:
            figures.append(plot_lr_coefficients(
                self.models['Logistic Regression'],
                self.feature_names
            ))

        if 'EBM' in self.models:
            figures.append(plot_ebm_shape_functions(self.models['EBM']))

        # Radar chart
        if all(k in self.results for k in ['accuracy', 'reliability', 'trust']):
            pillar_data = []
            for model_name in self.models.keys():
                pillar_data.append({
                    'Model': model_name,
                    'Accuracy': self.results['accuracy'][model_name]['auc'],
                    'Reliability': 1 - self.results['reliability'][model_name].get('ece', 0),
                    'Trust': self.results['trust'][model_name]['normalized_trust'],
                })
            figures.append(plot_radar_chart(pd.DataFrame(pillar_data)))

        logger.info(f"Generated {len(figures)} figures")
        return figures

    def generate_comprehensive_report(self) -> pd.DataFrame:
        """
        Generate comprehensive A-R-T metrics report.

        Returns:
            DataFrame with all metrics for all models
        """
        rows = []

        for model_name in self.models.keys():
            acc = self.results.get('accuracy', {}).get(model_name, {})
            rel = self.results.get('reliability', {}).get(model_name, {})
            trust = self.results.get('trust', {}).get(model_name, {})

            rows.append({
                'Model': model_name,
                # Accuracy
                'AUC': acc.get('auc'),
                'Gini': acc.get('gini'),
                'KS': acc.get('ks_statistic'),
                'Precision': acc.get('precision'),
                'Recall': acc.get('recall'),
                'F1': acc.get('f1'),
                # Reliability
                'ECE': rel.get('ece'),
                'MCE': rel.get('mce'),
                'Brier': rel.get('brier_score'),
                'PSI': rel.get('psi'),
                'AUC_Degradation': rel.get('auc_degradation'),
                # Trust
                'Trust_Score': trust.get('trust_score'),
                'Trust_Category': trust.get('model_category'),
                'Stability': trust.get('stability', {}).get('stability_score'),
            })

        report_df = pd.DataFrame(rows)

        # Save report
        report_path = RESULTS_DIR / 'art_comprehensive_report.csv'
        report_df.to_csv(report_path, index=False)
        logger.info(f"Saved report to {report_path}")

        return report_df

    def save_models(self) -> None:
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            safe_name = model_name.lower().replace(' ', '_').replace('-', '_')

            if model_name == 'FT-Transformer':
                # Save PyTorch model
                torch.save(
                    model.model.state_dict(),
                    MODELS_DIR / f"model_{safe_name}.pt"
                )
            else:
                # Save with pickle
                with open(MODELS_DIR / f"model_{safe_name}.pkl", 'wb') as f:
                    pickle.dump(model, f)

            logger.info(f"Saved {model_name}")

    def save_checkpoint(self, name: str = 'pipeline_checkpoint') -> None:
        """Save pipeline state as checkpoint."""
        checkpoint = {
            'config': self.config,
            'results': self.results,
            'feature_names': self.feature_names,
        }
        checkpoint_path = CHECKPOINTS_DIR / f'{name}.json'

        with open(checkpoint_path, 'w') as f:
            # Convert numpy to lists for JSON serialization
            json.dump(checkpoint, f, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def run_full_pipeline(
        self,
        data_path: str,
        target_col: str = 'bad',
        date_col: str = 'activation_date',
    ) -> pd.DataFrame:
        """
        Run the complete A-R-T evaluation pipeline.

        Args:
            data_path: Path to data CSV
            target_col: Target column name
            date_col: Date column for temporal split

        Returns:
            Comprehensive report DataFrame
        """
        logger.info("Starting full A-R-T evaluation pipeline")

        # Step 1: Load data
        self.load_and_prepare_data(data_path, target_col, date_col)

        # Step 2: Train models
        self.train_models(
            tune_hyperparameters=self.config.get('use_hyperparameter_tuning', True)
        )

        # Step 3: Evaluate all three pillars
        self.evaluate_accuracy()
        self.evaluate_reliability()
        self.evaluate_trust()

        # Step 4: Generate visualizations
        self.generate_visualizations()

        # Step 5: Save models and report
        self.save_models()
        report = self.generate_comprehensive_report()

        # Step 6: Checkpoint
        self.save_checkpoint()

        logger.info("Pipeline complete!")
        return report
