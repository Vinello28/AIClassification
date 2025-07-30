#!/usr/bin/env python
"""
Backward compatibility wrapper for the main classifier
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the refactored components
from src.ai_classification.core.classifier import AITextClassifier, quick_classify, classify_with_confidence
from src.ai_classification.core.config import CATEGORIES

# Make them available at module level for backward compatibility
__all__ = ['AITextClassifier', 'quick_classify', 'classify_with_confidence', 'CATEGORIES']

if __name__ == "__main__":
    # Quick test
    print("AI Classification system loaded successfully!")
    print(f"Available categories: {len(CATEGORIES)}")
    for k, v in CATEGORIES.items():
        print(f"  {k}: {v}")