"""
Feature Reduction Module.

Implements systematic correlation-based feature reduction for
Logistic Regression to satisfy statistical assumptions.

Features:
- Pearson correlation reduction for numerical variables
- Cramér's V reduction for categorical variables
- VIF-based multicollinearity reduction
- Encoding with one-hot and frequency strategies

Author: Lebohang Andile Skungwini
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from config import RANDOM_SEED, FEATURE_REDUCTION_CONFIG

logger = logging.getLogger(__name__)


class FeatureReductionWorkflow:
    """
    Systematic correlation-based feature reduction workflow for Logistic Regression.

    Removes redundant features while preserving predictive power with target.
    Designed to satisfy LR assumptions:
    1. No multicollinearity (VIF < 10)
    2. Removal of highly correlated feature pairs

    The workflow proceeds in steps:
    1. Numerical reduction (Pearson correlation)
    2. Categorical reduction (Cramér's V)
    3. Mixed-type reduction (correlation ratio η²)
    4. Categorical encoding
    5. VIF-based multicollinearity reduction
    6. Optional scaling
    """

    def __init__(
        self,
        target_col: str = 'bad',
        random_seed: int = RANDOM_SEED,
    ):
        """
        Initialize the workflow.

        Args:
            target_col: Name of the target column (binary)
            random_seed: Random seed for reproducibility
        """
        self.target_col = target_col
        self.random_seed = random_seed

        # Thresholds from config
        self.pearson_threshold = FEATURE_REDUCTION_CONFIG['pearson_threshold']
        self.cramers_threshold = FEATURE_REDUCTION_CONFIG['cramers_v_threshold']
        self.eta_threshold = FEATURE_REDUCTION_CONFIG['eta_threshold']
        self.vif_threshold = FEATURE_REDUCTION_CONFIG['vif_threshold']
        self.min_diff = FEATURE_REDUCTION_CONFIG['min_diff']

        # Track removed features for audit
        self.removed_features: List[Dict] = []
        self.final_features: Optional[List[str]] = None
        self.scaler: Optional[StandardScaler] = None

    def _calculate_cramers_v(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate Cramér's V for two categorical variables."""
        contingency = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(contingency)[0]
        n = len(x)
        min_dim = min(contingency.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * min_dim))

    def _calculate_correlation_ratio(
        self,
        categorical: pd.Series,
        numerical: pd.Series,
    ) -> float:
        """Calculate correlation ratio (η²) between categorical and numerical."""
        categories = categorical.dropna().unique()
        overall_mean = numerical.mean()
        overall_var = ((numerical - overall_mean) ** 2).sum()

        if overall_var == 0:
            return 0.0

        between_var = 0.0
        for cat in categories:
            cat_data = numerical[categorical == cat]
            if len(cat_data) > 0:
                between_var += len(cat_data) * (cat_data.mean() - overall_mean) ** 2

        return between_var / overall_var

    def _get_target_correlation(self, series: pd.Series, target: pd.Series) -> float:
        """Calculate correlation with binary target."""
        if series.dtype.name in ['category', 'object']:
            return self._calculate_cramers_v(series, target)
        else:
            return abs(series.corr(target))

    def step1_numerical_reduction(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Remove redundant numerical variables based on Pearson correlation.

        For pairs with |r| > threshold, keeps the variable with higher
        correlation with the target.

        Args:
            df: Input DataFrame
            threshold: Correlation threshold (default from config)

        Returns:
            DataFrame with reduced numerical features
        """
        threshold = threshold or self.pearson_threshold
        df = df.copy()

        # Get numerical columns
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols = [c for c in num_cols if c != self.target_col]

        if len(num_cols) < 2:
            return df

        # Calculate correlation matrix
        corr_matrix = df[num_cols].corr().abs()

        # Find highly correlated pairs
        to_remove = set()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if corr_matrix.iloc[i, j] > threshold:
                    col_i, col_j = num_cols[i], num_cols[j]

                    # Get target correlations
                    corr_i = self._get_target_correlation(
                        df[col_i], df[self.target_col]
                    )
                    corr_j = self._get_target_correlation(
                        df[col_j], df[self.target_col]
                    )

                    # Remove the one with lower target correlation
                    if corr_i >= corr_j:
                        to_remove.add(col_j)
                        reason = f"Correlated with {col_i} (r={corr_matrix.iloc[i,j]:.3f})"
                        self.removed_features.append({
                            'feature': col_j,
                            'step': 'numerical_reduction',
                            'reason': reason,
                        })
                    else:
                        to_remove.add(col_i)
                        reason = f"Correlated with {col_j} (r={corr_matrix.iloc[i,j]:.3f})"
                        self.removed_features.append({
                            'feature': col_i,
                            'step': 'numerical_reduction',
                            'reason': reason,
                        })

        if to_remove:
            df = df.drop(columns=list(to_remove))
            logger.info(f"Step 1: Removed {len(to_remove)} numerical features")

        return df

    def step2_categorical_reduction(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Remove redundant categorical variables based on Cramér's V.

        Args:
            df: Input DataFrame
            threshold: Cramér's V threshold (default from config)

        Returns:
            DataFrame with reduced categorical features
        """
        threshold = threshold or self.cramers_threshold
        df = df.copy()

        # Get categorical columns
        cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()

        if len(cat_cols) < 2:
            return df

        # Calculate pairwise Cramér's V
        to_remove = set()
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                col_i, col_j = cat_cols[i], cat_cols[j]

                if col_i in to_remove or col_j in to_remove:
                    continue

                cramers = self._calculate_cramers_v(df[col_i], df[col_j])

                if cramers > threshold:
                    # Get target correlations
                    corr_i = self._get_target_correlation(
                        df[col_i], df[self.target_col]
                    )
                    corr_j = self._get_target_correlation(
                        df[col_j], df[self.target_col]
                    )

                    if corr_i >= corr_j:
                        to_remove.add(col_j)
                        self.removed_features.append({
                            'feature': col_j,
                            'step': 'categorical_reduction',
                            'reason': f"Cramér's V with {col_i}: {cramers:.3f}",
                        })
                    else:
                        to_remove.add(col_i)
                        self.removed_features.append({
                            'feature': col_i,
                            'step': 'categorical_reduction',
                            'reason': f"Cramér's V with {col_j}: {cramers:.3f}",
                        })

        if to_remove:
            df = df.drop(columns=list(to_remove))
            logger.info(f"Step 2: Removed {len(to_remove)} categorical features")

        return df

    def step3_encode_categoricals(
        self,
        df: pd.DataFrame,
        high_cardinality_threshold: int = 10,
    ) -> pd.DataFrame:
        """
        Encode categorical variables.

        Strategy:
        - Low cardinality (≤ threshold): One-hot encoding
        - High cardinality (> threshold): Frequency encoding

        Args:
            df: Input DataFrame
            high_cardinality_threshold: Threshold for strategy selection

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        cat_cols = df.select_dtypes(include=['category', 'object']).columns
        cat_cols = [c for c in cat_cols if c != self.target_col]

        for col in cat_cols:
            n_unique = df[col].nunique()

            if n_unique <= high_cardinality_threshold:
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                # Frequency encoding
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[f"{col}_freq"] = df[col].map(freq_map)
                df = df.drop(columns=[col])

        logger.info(f"Step 3: Encoded {len(cat_cols)} categorical features")
        return df

    def step4_vif_reduction(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        max_iterations: int = 50,
    ) -> pd.DataFrame:
        """
        VIF-based multicollinearity reduction.

        Iteratively removes features with VIF > threshold.

        Args:
            df: Input DataFrame (should be encoded)
            threshold: VIF threshold (default from config)
            max_iterations: Maximum iterations to prevent infinite loops

        Returns:
            DataFrame with VIF-satisfying features
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        threshold = threshold or self.vif_threshold
        df = df.copy()

        # Get numerical columns (excluding target)
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols = [c for c in num_cols if c != self.target_col]

        iteration = 0
        while iteration < max_iterations:
            # Calculate VIF for all features
            X = df[num_cols].dropna()
            if len(X.columns) < 2:
                break

            vif_data = []
            for i, col in enumerate(X.columns):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_data.append({'feature': col, 'VIF': vif})
                except Exception:
                    vif_data.append({'feature': col, 'VIF': np.inf})

            vif_df = pd.DataFrame(vif_data)

            # Find max VIF
            max_vif = vif_df['VIF'].max()
            if max_vif <= threshold:
                break

            # Remove feature with highest VIF
            worst_feature = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
            num_cols.remove(worst_feature)
            df = df.drop(columns=[worst_feature])

            self.removed_features.append({
                'feature': worst_feature,
                'step': 'vif_reduction',
                'reason': f"VIF = {max_vif:.2f} > {threshold}",
            })

            iteration += 1

        logger.info(f"Step 4: VIF reduction complete after {iteration} iterations")
        return df

    def step5_scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()

        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        num_cols = [c for c in num_cols if c != self.target_col]

        self.scaler = StandardScaler()
        df[num_cols] = self.scaler.fit_transform(df[num_cols])

        logger.info(f"Step 5: Scaled {len(num_cols)} numerical features")
        return df

    def run_full_workflow(
        self,
        df: pd.DataFrame,
        scale_features: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete feature reduction workflow.

        Args:
            df: Input DataFrame
            scale_features: Whether to apply scaling at the end

        Returns:
            Reduced and processed DataFrame
        """
        logger.info("Starting feature reduction workflow")
        initial_cols = len(df.columns)

        # Step 1: Numerical correlation reduction
        df = self.step1_numerical_reduction(df)

        # Step 2: Categorical association reduction
        df = self.step2_categorical_reduction(df)

        # Step 3: Encode categoricals
        df = self.step3_encode_categoricals(df)

        # Step 4: VIF reduction
        df = self.step4_vif_reduction(df)

        # Step 5: Scale features
        if scale_features:
            df = self.step5_scale_features(df)

        # Store final features
        self.final_features = [
            c for c in df.columns if c != self.target_col
        ]

        final_cols = len(df.columns)
        logger.info(
            f"Workflow complete: {initial_cols} → {final_cols} columns "
            f"({len(self.removed_features)} removed)"
        )

        return df

    def get_removal_report(self) -> pd.DataFrame:
        """Return DataFrame with removal audit trail."""
        return pd.DataFrame(self.removed_features)

    def apply_to_new_data(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the same reduction to new data (validation/test).

        Uses the final_features list from training.

        Args:
            df: New DataFrame to transform

        Returns:
            Transformed DataFrame with same features as training
        """
        if self.final_features is None:
            raise ValueError("Must run workflow on training data first")

        df = df.copy()

        # Keep only final features plus target
        keep_cols = self.final_features + [self.target_col]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        # Apply scaling if fitted
        if self.scaler is not None:
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            num_cols = [c for c in num_cols if c != self.target_col]
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df
