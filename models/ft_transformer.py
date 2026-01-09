"""
FT-Transformer Model.

Feature Tokenizer + Transformer for tabular data.
Deep learning architecture using self-attention mechanisms.

Author: Lebohang Andile Skungwini
"""

import logging
from typing import List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import RANDOM_SEED, CONFIG, TORCH_GPU_AVAILABLE
from .base import BaseModel

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer Architecture.

    Converts each feature into a token embedding, applies
    transformer layers with self-attention, and produces
    binary classification output.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        d_ffn_factor: float = 4/3,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # Feature tokenizer: one embedding per feature
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])

        # [CLS] token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer layers
        d_ffn = int(d_model * d_ffn_factor)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for emb in self.feature_embeddings:
            nn.init.xavier_uniform_(emb.weight)
            nn.init.zeros_(emb.bias)
        nn.init.xavier_uniform_(self.output_layer[-1].weight)
        nn.init.zeros_(self.output_layer[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_features)

        Returns:
            Logits of shape (batch_size,)
        """
        batch_size = x.shape[0]

        # Tokenize each feature
        tokens = []
        for i, emb in enumerate(self.feature_embeddings):
            feature_token = emb(x[:, i:i+1])  # (batch, d_model)
            tokens.append(feature_token)

        # Stack tokens: (batch, n_features, d_model)
        tokens = torch.stack(tokens, dim=1)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Apply transformer
        output = self.transformer(tokens)

        # Use [CLS] token for classification
        cls_output = output[:, 0, :]
        logits = self.output_layer(cls_output).squeeze(-1)

        return logits


class FTTransformerModel(BaseModel):
    """
    Wrapper for FT-Transformer with training loop.

    Attributes:
        n_features: Number of input features
        device: 'cuda' or 'cpu'
        random_state: Random seed
        n_trials: Number of Optuna trials
        model: FTTransformer neural network
        best_params: Best hyperparameters
    """

    name = "FT-Transformer"

    def __init__(
        self,
        n_features: int = None,
        device: str = None,
        random_state: int = RANDOM_SEED,
        n_trials: int = None,
    ):
        """
        Initialize FT-Transformer model.

        Args:
            n_features: Number of input features (can be set during training)
            device: 'cuda' or 'cpu' (auto-detect if None)
            random_state: Random seed
            n_trials: Number of Optuna trials
        """
        self.n_features = n_features
        self.device = device or ('cuda' if TORCH_GPU_AVAILABLE else 'cpu')
        self.random_state = random_state
        self.n_trials = n_trials or CONFIG.get('tuning_trials_ftt', 20)
        self.model: Optional[FTTransformer] = None
        self.best_params: Optional[dict] = None

        # Set seeds
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def _objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Optuna objective for hyperparameter optimization."""
        n_features = X_train.shape[1]

        params = {
            'd_model': trial.suggest_categorical('d_model', [64, 128, 192, 256]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8]),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        }

        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])

        # Create model
        model = FTTransformer(n_features=n_features, **params).to(self.device)

        # Training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        train_loader = self._create_dataloader(X_train, y_train, batch_size)

        model.train()
        for epoch in range(20):  # Quick evaluation
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            logits = model(X_val_t)
            probs = torch.sigmoid(logits).cpu().numpy()

        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_val, probs)

        return auc

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 512,
        lr: float = 1e-4,
        tune_hyperparameters: bool = True,
    ) -> None:
        """
        Train FT-Transformer.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            tune_hyperparameters: Whether to tune hyperparameters
        """
        logger.info(f"Training {self.name} on {self.device}")

        self.n_features = X_train.shape[1]

        if tune_hyperparameters and X_val is not None and y_val is not None:
            logger.info(f"Running {self.n_trials} optimization trials")

            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(
                    trial, X_train, y_train, X_val, y_val
                ),
                n_trials=self.n_trials,
                show_progress_bar=True,
            )

            self.best_params = study.best_params
            logger.info(f"Best AUC: {study.best_value:.4f}")
            logger.info(f"Best params: {self.best_params}")

            # Extract architecture params
            arch_params = {
                k: v for k, v in self.best_params.items()
                if k in ['d_model', 'n_heads', 'n_layers', 'dropout']
            }
            lr = self.best_params.get('learning_rate', lr)
            batch_size = self.best_params.get('batch_size', batch_size)
        else:
            arch_params = {
                'd_model': 192,
                'n_heads': 8,
                'n_layers': 3,
                'dropout': 0.1,
            }
            self.best_params = arch_params
            logger.info("Using default hyperparameters")

        # Create final model
        self.model = FTTransformer(
            n_features=self.n_features,
            **arch_params,
        ).to(self.device)

        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )

        train_loader = self._create_dataloader(X_train, y_train, batch_size)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, batch_size, shuffle=False)
        else:
            val_loader = None

        # Training loop with early stopping
        best_val_auc = 0
        patience = 10
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            if val_loader is not None:
                self.model.eval()
                all_probs = []
                all_labels = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        logits = self.model(X_batch)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        all_probs.extend(probs)
                        all_labels.extend(y_batch.numpy())

                from sklearn.metrics import roc_auc_score
                val_auc = roc_auc_score(all_labels, all_probs)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: loss={train_loss/len(train_loader):.4f}, "
                        f"val_auc={val_auc:.4f}"
                    )

        logger.info(f"{self.name} training complete (best val AUC: {best_val_auc:.4f})")

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Predict probability of positive class."""
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.eval()
        all_probs = []

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)

        return np.array(all_probs)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def predict_logits(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Predict raw logits (for temperature scaling).

        Args:
            X: Feature matrix

        Returns:
            Array of raw logits before sigmoid
        """
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.eval()
        all_logits = []

        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                logits = self.model(X_batch).cpu().numpy()
                all_logits.extend(logits)

        return np.array(all_logits)
