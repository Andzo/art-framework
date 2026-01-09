#!/usr/bin/env python
"""
A-R-T Framework CLI Entry Point.

Command-line interface for running the Accuracy-Reliability-Trust
evaluation pipeline.

Usage:
    python run_evaluation.py --data data.csv --target bad --output results/

Author: Lebohang Andile Skungwini
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG, RESULTS_DIR
from pipeline import ARTEvaluationPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='A-R-T Framework: Accuracy-Reliability-Trust Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline:
    python run_evaluation.py --data data.csv --target bad

  Skip hyperparameter tuning:
    python run_evaluation.py --data data.csv --skip-tuning

  Specific models only:
    python run_evaluation.py --data data.csv --models lr,xgb

  Load pre-trained models:
    python run_evaluation.py --data data.csv --load-models
        """
    )

    # Required arguments
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    # Optional arguments
    parser.add_argument(
        '--target', '-t',
        type=str,
        default='bad',
        help='Target column name (default: bad)'
    )

    parser.add_argument(
        '--date-col',
        type=str,
        default='activation_date',
        help='Date column for temporal split (default: activation_date)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--models',
        type=str,
        default='lr,xgb,ebm,ftt',
        help='Comma-separated list of models to train (lr,xgb,ebm,ftt)'
    )

    parser.add_argument(
        '--skip-tuning',
        action='store_true',
        help='Skip hyperparameter optimization'
    )

    parser.add_argument(
        '--load-models',
        action='store_true',
        help='Load pre-trained models instead of training'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    parser.add_argument(
        '--random-split',
        action='store_true',
        help='Use random split instead of temporal split'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Update config
    config = CONFIG.copy()
    config['random_seed'] = args.seed
    config['use_hyperparameter_tuning'] = not args.skip_tuning

    if args.output:
        config['results_dir'] = args.output
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Parse model selection
    model_map = {
        'lr': 'Logistic Regression',
        'xgb': 'XGBoost',
        'ebm': 'EBM',
        'ftt': 'FT-Transformer',
    }
    selected_models = [
        model_map[m.strip().lower()]
        for m in args.models.split(',')
        if m.strip().lower() in model_map
    ]

    logger.info("=" * 60)
    logger.info("A-R-T Framework: Accuracy-Reliability-Trust Evaluation")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Models: {', '.join(selected_models)}")
    logger.info(f"Tuning: {'Enabled' if not args.skip_tuning else 'Disabled'}")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = ARTEvaluationPipeline(config)

    try:
        # Run pipeline
        report = pipeline.run_full_pipeline(
            data_path=args.data,
            target_col=args.target,
            date_col=args.date_col if not args.random_split else None,
        )

        # Display summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print("\nA-R-T Summary:")
        print(report[['Model', 'AUC', 'ECE', 'Trust_Score']].to_string(index=False))
        print("\n" + "=" * 60)
        print(f"Results saved to: {config['results_dir']}")
        print(f"Figures saved to: {config['figures_dir']}")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == '__main__':
    main()
