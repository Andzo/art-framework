"""
FT-Transformer Training Script for Google Colab.

This script trains the FT-Transformer model on Google Colab using
data stored in Google Drive, then saves the trained model back to Drive.

Upload this script to Colab or copy the contents into a notebook.

Instructions:
1. Upload this script to Colab or copy into a notebook cell
2. Mount your Google Drive
3. Update DATA_DIR and OUTPUT_DIR paths
4. Run all cells

Author: Lebohang Andile Skungwini
"""

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Path to your data folder on Google Drive (containing train_df.csv, val_df.csv, test_df.csv)
DATA_DIR = "/content/drive/MyDrive/UKZN_Masters/art_source_data"

# Path to save the trained model on Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/UKZN_Masters/trained_models"

# Training configuration
ENABLE_TUNING = False  # Set True for hyperparameter tuning (takes longer)
N_TUNING_TRIALS = 20   # Number of Optuna trials if tuning enabled
RANDOM_SEED = 42

# =============================================================================
# SETUP - Mount Drive and Install Dependencies
# =============================================================================

def setup_colab():
    """Mount Google Drive and install required packages."""
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Install required packages
    import subprocess
    subprocess.run(['pip', 'install', '-q', 'optuna', 'imbalanced-learn', 'shap'], check=True)
    print("Setup complete!")

# Uncomment the line below when running in Colab:
# setup_colab()

# =============================================================================
# IMPORTS
# =============================================================================

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# Set seeds
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


# =============================================================================
# FT-TRANSFORMER MODEL DEFINITION
# =============================================================================

class FeatureTokenizer(nn.Module):
    """Tokenizes numerical features into embeddings."""
    
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Linear(1, d_model)
        self.feature_biases = nn.Parameter(torch.zeros(n_features, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        tokens = self.embeddings(x) + self.feature_biases
        return tokens


class FTTransformer(nn.Module):
    """Feature Tokenizer Transformer for tabular data."""
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        
        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Tokenize features
        tokens = self.tokenizer(x)  # (batch, n_features, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch, n_features+1, d_model)
        
        # Transformer forward
        encoded = self.transformer(tokens)
        
        # Use CLS token for classification
        cls_output = encoded[:, 0]
        logits = self.head(cls_output)
        
        return logits.squeeze(-1)


class FTTransformerModel:
    """Wrapper class for FT-Transformer with training utilities."""
    
    def __init__(self, n_features: int, random_state: int = 42, n_trials: int = 20):
        self.n_features = n_features
        self.random_state = random_state
        self.n_trials = n_trials
        self.device = DEVICE
        self.model = None
        self.best_params = None
        
    def _create_dataloaders(self, X_train, y_train, X_val, y_val, batch_size):
        """Create PyTorch DataLoaders."""
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def _train_epoch(self, model, train_loader, optimizer, criterion):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate(self, model, val_loader, criterion):
        """Validate model."""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        return total_loss / len(val_loader), auc
    
    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=False):
        """Train the FT-Transformer model."""
        logger.info("Training FT-Transformer")
        
        if tune_hyperparameters:
            self._tune_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            # Default parameters
            self.best_params = {
                'd_model': 64,
                'n_heads': 4,
                'n_layers': 2,
                'dropout': 0.1,
                'learning_rate': 0.001,
                'batch_size': 256,
            }
            logger.info("Using default hyperparameters")
        
        # Create model
        arch_params = {k: v for k, v in self.best_params.items() 
                      if k in ['d_model', 'n_heads', 'n_layers', 'dropout']}
        self.model = FTTransformer(n_features=self.n_features, **arch_params).to(self.device)
        
        # Training setup
        batch_size = self.best_params.get('batch_size', 256)
        lr = self.best_params.get('learning_rate', 0.001)
        
        train_loader, val_loader = self._create_dataloaders(X_train, y_train, X_val, y_val, batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        n_epochs = 50
        best_auc = 0
        patience_counter = 0
        best_state = None
        
        for epoch in range(n_epochs):
            train_loss = self._train_epoch(self.model, train_loader, optimizer, criterion)
            val_loss, val_auc = self._validate(self.model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}")
            
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_state:
            self.model.load_state_dict(best_state)
        
        logger.info(f"FT-Transformer training complete. Best validation AUC: {best_auc:.4f}")
    
    def _tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Hyperparameter tuning with Optuna."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {
                'd_model': trial.suggest_categorical('d_model', [32, 64, 128]),
                'n_heads': trial.suggest_categorical('n_heads', [2, 4, 8]),
                'n_layers': trial.suggest_int('n_layers', 1, 4),
                'dropout': trial.suggest_float('dropout', 0.05, 0.3),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
            }
            
            model = FTTransformer(
                n_features=self.n_features,
                d_model=params['d_model'],
                n_heads=params['n_heads'],
                n_layers=params['n_layers'],
                dropout=params['dropout'],
            ).to(self.device)
            
            train_loader, val_loader = self._create_dataloaders(
                X_train, y_train, X_val, y_val, params['batch_size']
            )
            
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.BCEWithLogitsLoss()
            
            for epoch in range(20):  # Quick training for tuning
                self._train_epoch(model, train_loader, optimizer, criterion)
                _, val_auc = self._validate(model, val_loader, criterion)
                
                trial.report(val_auc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return val_auc
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        logger.info(f"Best hyperparameters: {self.best_params}")
    
    def predict_proba(self, X):
        """Get probability predictions."""
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits)
        
        return probs.cpu().numpy()
    
    def predict(self, X, threshold=0.5):
        """Get class predictions."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load train, val, test CSVs from Google Drive."""
    data_path = Path(data_dir)
    
    logger.info(f"Loading data from {data_path}")
    
    train_df = pd.read_csv(data_path / 'train_df.csv')
    val_df = pd.read_csv(data_path / 'val_df.csv')
    test_df = pd.read_csv(data_path / 'test_df.csv')
    
    logger.info(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


def prepare_features(df: pd.DataFrame, target_col='bad', exclude_cols=None, 
                     scaler_params=None, fit_scaler=False):
    """Prepare features with encoding and scaling."""
    df = df.copy()
    exclude_cols = exclude_cols or ['activation_date']
    
    y = df[target_col].values
    drop_cols = [target_col] + [c for c in exclude_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Handle missing values
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) > 0 else 'Unknown')
    
    # Encode categoricals
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map).fillna(0)
            df = df.rename(columns={col: f"{col}_freq"})
    
    feature_names = df.columns.tolist()
    
    # Z-score scaling
    if fit_scaler:
        scaler_params = {'mean': df.mean().to_dict(), 'std': df.std().to_dict()}
    
    if scaler_params:
        for col in df.columns:
            if col in scaler_params['mean']:
                mean = scaler_params['mean'][col]
                std = scaler_params['std'].get(col, 1.0)
                if std > 0:
                    df[col] = (df[col] - mean) / std
    
    X = df.values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y, feature_names, scaler_params


def apply_smote_enn(X, y):
    """Apply SMOTE-ENN resampling."""
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    
    logger.info(f"Before SMOTE-ENN: {len(y):,} samples, bad rate: {y.mean():.2%}")
    
    smote_enn = SMOTEENN(
        smote=SMOTE(k_neighbors=5, random_state=RANDOM_SEED),
        enn=EditedNearestNeighbours(n_neighbors=3, sampling_strategy='all'),
        random_state=RANDOM_SEED,
    )
    
    X_res, y_res = smote_enn.fit_resample(X, y)
    logger.info(f"After SMOTE-ENN: {len(y_res):,} samples, bad rate: {y_res.mean():.2%}")
    
    return X_res, y_res


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def train_and_save():
    """Main training function."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FT-TRANSFORMER TRAINING (Google Colab)")
    logger.info("=" * 60)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Tuning enabled: {ENABLE_TUNING}")
    logger.info(f"Device: {DEVICE}")
    
    # Load data
    logger.info("\n[1/5] Loading data...")
    data = load_data(DATA_DIR)
    
    # Prepare features
    logger.info("\n[2/5] Preparing features...")
    X_train, y_train, feature_names, scaler_params = prepare_features(data['train'], fit_scaler=True)
    X_val, y_val, _, _ = prepare_features(data['val'], scaler_params=scaler_params)
    X_test, y_test, _, _ = prepare_features(data['test'], scaler_params=scaler_params)
    
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Apply SMOTE-ENN
    logger.info("\n[3/5] Applying SMOTE-ENN...")
    X_train_res, y_train_res = apply_smote_enn(X_train, y_train)
    
    # Train model
    logger.info("\n[4/5] Training FT-Transformer...")
    model = FTTransformerModel(
        n_features=X_train_res.shape[1],
        random_state=RANDOM_SEED,
        n_trials=N_TUNING_TRIALS,
    )
    model.train(X_train_res, y_train_res, X_val, y_val, tune_hyperparameters=ENABLE_TUNING)
    
    # Evaluate
    logger.info("\n[5/5] Evaluating on test set...")
    y_pred_proba = model.predict_proba(X_test)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Test AUC: {test_auc:.4f}")
    
    # Save model
    model_path = output_dir / 'ft_transformer.pt'
    torch.save(model.model.state_dict(), model_path)
    logger.info(f"Model saved to: {model_path}")
    
    # Save config
    config = {
        'n_features': model.n_features,
        'best_params': model.best_params,
        'test_auc': float(test_auc),
        'training_date': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
    }
    config_path = output_dir / 'ft_transformer_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to: {config_path}")
    
    # Save scaler params for local use
    scaler_path = output_dir / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f)
    logger.info(f"Scaler params saved to: {scaler_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Download these files to your local results/models folder:")
    logger.info(f"  - {model_path}")
    logger.info(f"  - {config_path}")
    
    return model, test_auc


# =============================================================================
# RUN TRAINING
# =============================================================================

if __name__ == '__main__':
    train_and_save()
