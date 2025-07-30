"""
AI Classification Package

A text classification system for AI-related content.
"""

__version__ = "1.0.0"
__author__ = "AIClassification Team"

# Import config directly (no torch dependency)
from .core.config import CATEGORIES

# Lazy import for classifier to avoid torch dependency at import time
def get_classifier(*args, **kwargs):
    """Lazy loader for AITextClassifier to avoid torch import at package level"""
    from .core.classifier import AITextClassifier
    return AITextClassifier(*args, **kwargs)

# Make categories available directly
__all__ = ["get_classifier", "CATEGORIES"]