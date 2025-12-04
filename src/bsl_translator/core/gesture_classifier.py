"""
BSL Translator - Gesture Classifier Module
ML model loading and gesture classification.
"""
import os
import pickle
import numpy as np
from typing import Optional
import logging

from .config import ModelConfig

logger = logging.getLogger(__name__)


class GestureClassifier:
    """BSL gesture classification using trained sklearn model."""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[ModelConfig] = None):
        """
        Initialize gesture classifier.
        
        Args:
            model_path: Explicit path to model file
            config: Model configuration
        """
        if config is None:
            config = ModelConfig()
        
        self.config = config
        self.model = None
        self.model_path = None
        
        # Load model
        if model_path:
            self._load_model(model_path)
        else:
            self._find_and_load_model()
    
    def _load_model(self, path: str) -> bool:
        """
        Load model from specific path.
        
        Args:
            path: Path to model file
            
        Returns:
            True if successful
        """
        try:
            with open(path, 'rb') as f:
                model_dict = pickle.load(f)
                
            if 'model' in model_dict:
                self.model = model_dict['model']
                self.model_path = path
                logger.info(f"Model loaded from {path}")
                return True
            else:
                logger.warning(f"Model file at {path} doesn't contain 'model' key")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return False
    
    def _find_and_load_model(self) -> None:
        """Find and load model from default paths."""
        # Build search paths
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        current_dir = os.getcwd()
        
        search_paths = self.config.default_paths + [
            os.path.join(script_dir, 'model.p'),
            os.path.join(script_dir, 'models', 'bsl_classifier.pkl'),
            os.path.join(current_dir, 'model.p'),
            os.path.join(current_dir, 'models', 'bsl_classifier.pkl'),
            os.path.join(os.path.dirname(current_dir), 'model.p'),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                if self._load_model(path):
                    return
        
        # Model not found
        raise FileNotFoundError(
            f"Model file not found. Searched in:\n" + 
            "\n".join(f"  - {p}" for p in search_paths[:5])
        )
    
    def predict(self, features: np.ndarray) -> Optional[str]:
        """
        Predict gesture from feature array.
        
        Args:
            features: Feature array of shape (1, 84)
            
        Returns:
            Predicted letter (uppercase) or None
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
            
        if features is None or features.size == 0:
            return None
        
        try:
            # Ensure correct feature count
            expected = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else self.config.expected_features
            actual = features.shape[1]
            
            if actual != expected:
                # Pad or truncate
                if actual < expected:
                    padding = np.zeros((1, expected - actual))
                    features = np.hstack((features, padding))
                else:
                    features = features[:, :expected]
            
            prediction = self.model.predict(features)
            return str(prediction[0]).upper()
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        return self.model is not None
