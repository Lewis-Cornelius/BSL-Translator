"""
Tests for WordPredictor module.
"""
import pytest
from unittest.mock import patch, Mock

from bsl_translator.core import WordPredictor


class TestWordPredictor:
    """Test suite for WordPredictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create a WordPredictor with mock dictionary."""
        with patch('nltk.download'):
            with patch('bsl_translator.core.word_predictor.words.words') as mock_words:
                # Mock word list
                mock_words.return_value = [
                    'HELLO', 'HELP', 'WORLD', 'WORD', 'WORK',
                    'THANK', 'THANKS', 'PLEASE', 'YES', 'NO'
                ]
                predictor = WordPredictor()
                # Override dictionary for testing
                predictor.dictionary = set([
                    'HELLO', 'HELP', 'WORLD', 'WORD', 'WORK',
                    'THANK', 'THANKS', 'PLEASE', 'YES', 'NO'
                ])
                return predictor
    
    def test_get_suggestions_returns_close_matches(self, predictor):
        """Test get_suggestions returns similar words."""
        suggestions = predictor.get_suggestions('HEL', num=3)
        
        assert 'HELLO' in suggestions or 'HELP' in suggestions
        assert len(suggestions) <= 3
    
    def test_get_suggestions_with_empty_fragment(self, predictor):
        """Test get_suggestions with empty string."""
        suggestions = predictor.get_suggestions('', num=3)
        
        assert suggestions == []
    
    def test_get_suggestions_limits_results(self, predictor):
        """Test get_suggestions respects num parameter."""
        suggestions = predictor.get_suggestions('WOR', num=2)
        
        assert len(suggestions) <= 2
    
    def test_autocorrect_returns_exact_match(self, predictor):
        """Test autocorrect returns word if in dictionary."""
        result = predictor.autocorrect('HELLO')
        
        assert result == 'HELLO'
    
    def test_autocorrect_fixes_typo(self, predictor):
        """Test autocorrect suggests correction for typo."""
        result = predictor.autocorrect('HELO')  # Missing L
        
        # Should suggest HELLO
        assert result == 'HELLO'
    
    def test_autocorrect_returns_original_if_no_match(self, predictor):
        """Test autocorrect returns original if no good match."""
        result = predictor.autocorrect('XYZABC')
        
        # Should return original
        assert result == 'XYZABC'
    
    def test_autocorrect_case_insensitive(self, predictor):
        """Test autocorrect handles lowercase input."""
        result = predictor.autocorrect('hello')
        
        # Should return uppercase
        assert result == 'HELLO'
    
    def test_fallback_dictionary_on_nltk_failure(self):
        """Test fallback dictionary when NLTK fails."""
        with patch('nltk.download', side_effect=Exception("NLTK error")):
            predictor = WordPredictor()
            
            # Should have fallback dictionary
            assert len(predictor.dictionary) > 0
            assert 'HELLO' in predictor.dictionary
