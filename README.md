# A-R-T Framework

## Overview

This framework operationalises the A-R-T evaluation methodology presented in *"An Accuracy-Reliability-Trust Evaluation of ML models for Post-Activation Credit Risk in the South African Telecommunications Sector"*.

### Three-Pillar Evaluation

| Pillar | Metrics | Purpose |
|--------|---------|---------|
| **Accuracy** | AUC, Gini, KS Statistic, Top-Decile Capture | Discrimination performance |
| **Reliability** | ECE, Brier Score, PSI, OOT Degradation | Calibration & temporal stability |
| **Trust** | Compliance Rubric (0-9), Stability, Faithfulness | Regulatory explainability |

### Supported Models

- **Logistic Regression** (Elastic Net with Optuna tuning)
- **XGBoost** (with GPU acceleration)
- **Explainable Boosting Machine** (EBM - glass-box)
- **FT-Transformer** (Feature Tokenizer + Transformer)

## Installation

```bash
# Clone the repository
git clone https://github.com/[username]/art-framework.git
cd art-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Full Pipeline Evaluation

```bash
python run_evaluation.py \
    --data path/to/data.csv \
    --target bad \
    --date-col activation_date \
    --output results/
```

### Python API

```python
from pipeline import ARTEvaluationPipeline
from config import CONFIG

# Initialize pipeline
pipeline = ARTEvaluationPipeline(CONFIG)

# Load and prepare data
data = pipeline.load_and_prepare_data(
    data_path='data.csv',
    target_col='bad',
    date_col='activation_date',
    use_temporal_split=True
)

# Train all models
pipeline.train_models(data)

# Evaluate all three pillars
pipeline.evaluate_accuracy(data)
pipeline.evaluate_reliability(data)
pipeline.evaluate_trust(data)

# Generate comprehensive report
pipeline.generate_comprehensive_report()
```

### Individual Module Usage

```python
# Preprocessing
from preprocessing import load_raw_data, preprocess_data, SMOTEENNResampler

df = load_raw_data('data.csv')
df = preprocess_data(df, target_col='bad')

# Train a single model
from models import XGBoostModel

model = XGBoostModel()
model.train(X_train, y_train, X_val, y_val)
predictions = model.predict_proba(X_test)

# Evaluate accuracy
from evaluation import AccuracyEvaluator

metrics = AccuracyEvaluator.calculate_metrics(y_test, predictions)
print(f"AUC: {metrics['auc']:.4f}, Gini: {metrics['gini']:.4f}")

# Generate visualizations
from visualization import plot_roc_curve, plot_calibration_curve

plot_roc_curve(y_test, predictions, 'XGBoost', save_path='figures/')
```

## Repository Structure

```
art_framework_repo/
├── config.py                 # Global configuration
├── pipeline.py               # Main ARTEvaluationPipeline
├── run_evaluation.py         # CLI entry point
├── requirements.txt          # Dependencies
│
├── preprocessing/
│   ├── data_loader.py        # Data loading and type conversion
│   ├── feature_engineering.py # Feature transformations
│   ├── resampling.py         # SMOTE-ENN for class imbalance
│   └── feature_reduction.py  # VIF/correlation reduction for LR
│
├── models/
│   ├── base.py               # Abstract base model class
│   ├── logistic_regression.py
│   ├── xgboost_model.py
│   ├── ebm.py
│   └── ft_transformer.py
│
├── evaluation/
│   ├── accuracy.py           # Discrimination metrics
│   ├── reliability.py        # Calibration & stability
│   └── trust.py              # Explainability & compliance
│
└── visualization/
    ├── eda_plots.py          # Exploratory analysis
    ├── model_plots.py        # Model-specific plots
    └── comparative_plots.py  # Cross-model comparisons
```

## Configuration

Edit `config.py` to customize:

```python
CONFIG = {
    'random_seed': 42,
    'use_smote_enn': True,
    'use_hyperparameter_tuning': True,
    'tuning_trials_xgb': 50,
    # ... see config.py for full options
}
```

## CLI Options

```bash
python run_evaluation.py --help

Options:
  --data PATH           Path to input CSV file
  --target COL          Target column name (default: 'bad')
  --date-col COL        Date column for temporal split
  --output DIR          Output directory for results
  --models LIST         Comma-separated model list (lr,xgb,ebm,ftt)
  --skip-tuning         Skip hyperparameter optimization
  --load-models         Load pre-trained models instead of training
  --resume              Resume from checkpoints if available
```

## Output

The framework generates:

- **Results CSVs**: Comprehensive metrics, pillar scores, rankings
- **Figures**: ROC curves, calibration plots, SHAP summaries, radar charts
- **Reports**: Markdown summary with interpretation guidance
- **Checkpoints**: Resumable pipeline state

## Citation

If you use this framework in your research, please cite:

```bibtex
@mastersthesis{skungwini2025art,
  title={An Accuracy-Reliability-Trust Evaluation of ML models for 
         Post-Activation Credit Risk in the South African 
         Telecommunications Sector},
  author={Skungwini, Lebohang Andile},
  year={2025},
  school={University of KwaZulu-Natal}
}
```

## License

This project is provided for academic purposes. See LICENSE for details.

## Author

**Lebohang Andile Skungwini**  
MSc Data Science, University of KwaZulu-Natal
