"""
Evaluation module for A-R-T Framework.

Contains metrics and evaluators for all three pillars:
- Accuracy: Discrimination metrics (AUC, Gini, KS, DeLong test)
- Reliability: Calibration (ECE, Brier) and temporal stability (PSI)
- Trust: Regulatory compliance rubric and technical explainability
"""

from .accuracy import AccuracyEvaluator
from .reliability import ReliabilityEvaluator
from .trust import TrustEvaluator

__all__ = [
    'AccuracyEvaluator',
    'ReliabilityEvaluator',
    'TrustEvaluator',
]
