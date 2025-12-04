"""
BSL Translator - Real-Time British Sign Language Recognition
Main package for BSL gesture translation.
"""

__version__ = '1.0.0'
__author__ = 'Lewis Cornelius'

# Import core modules for convenience
from .core import (
    Config,
    default_config,
    HandDetector,
    GestureClassifier,
    WordPredictor
)

__all__ = [
    '__version__',
    '__author__',
    'Config',
    'default_config',
    'HandDetector',
    'GestureClassifier',
    'WordPredictor',
]
