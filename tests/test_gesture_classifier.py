"""
Tests for GestureClassifier module.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pickle

from bsl_translator.core import GestureClassifier, ModelConfig


class TestGestureClassifier:
    """Test suite for GestureClassifier."""
    
    def test_classifier_initialization_with_valid_model(self):
        """Test classifier initializes with valid model path."""
        # Mock model file
        mock_model = Mock()
        mock_model.n_features_in_ = 84
        mock_model.predict = Mock(return_value=['A'])
        
        model_dict = {'model': mock_model}
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=model_dict):
                with patch('os.path.exists', return_value=True):
                    classifier = GestureClassifier(model_path='test_model.p')
                    
                    assert classifier.is_loaded()
                    assert classifier.model == mock_model
    
    def test_predict_returns_uppercase_letter(self):
        """Test prediction returns uppercase letter."""
        mock_model = Mock()
        mock_model.n_features_in_ = 84
        mock_model.predict = Mock(return_value=['a'])  # lowercase
        
        model_dict = {'model': mock_model}
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=model_dict):
                with patch('os.path.exists', return_value=True):
                    classifier = GestureClassifier(model_path='test_model.p')
                    
                    features = np.zeros((1, 84))
                    result = classifier.predict(features)
                    
                    assert result == 'A'  # Should be uppercase
    
    def test_predict_with_none_features(self):
        """Test prediction handles None features gracefully."""
        mock_model = Mock()
        model_dict = {'model': mock_model}
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=model_dict):
                with patch('os.path.exists', return_value=True):
                    classifier = GestureClassifier(model_path='test_model.p')
                    
                    result = classifier.predict(None)
                    
                    assert result is None
    
    def test_predict_pads_insufficient_features(self):
        """Test prediction pads features if too few."""
        mock_model = Mock()
        mock_model.n_features_in_ = 84
        mock_model.predict = Mock(return_value=['B'])
        
        model_dict = {'model': mock_model}
        
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value=model_dict):
                with patch('os.path.exists', return_value=True):
                    classifier = GestureClassifier(model_path='test_model.p')
                    
                    # Too few features
                    features = np.zeros((1, 40))
                    result = classifier.predict(features)
                    
                    # Should still work (padded internally)
                    assert result == 'B'
                    
                    # Check that predict was called with correct shape
                    call_args = mock_model.predict.call_args[0][0]
                    assert call_args.shape[1] == 84
    
    def test_is_loaded_returns_false_when_no_model(self):
        """Test is_loaded returns False when model not loaded."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                classifier = GestureClassifier(model_path='nonexistent.p')
    
    def test_model_file_without_model_key(self):
        """Test handles model file without 'model' key."""
        with patch('builtins.open', mock_open()):
            with patch('pickle.load', return_value={}):  # No 'model' key
                with patch('os.path.exists', return_value=True):
                    with pytest.raises(FileNotFoundError):
                        classifier = GestureClassifier(model_path='test.p')
