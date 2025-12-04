"""
Tests for GestureTracker from demo_standalone.
"""
import pytest
import time
from unittest.mock import Mock

import sys
sys.path.insert(0, 'src')

from demo_standalone import GestureTracker
from bsl_translator.core import Config


class TestGestureTracker:
    """Test suite for GestureTracker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config()
    
    @pytest.fixture
    def tracker(self, config):
        """Create GestureTracker instance."""
        return GestureTracker(config)
    
    def test_initialization(self, tracker, config):
        """Test tracker initializes correctly."""
        assert tracker.gesture_buffer == []
        assert tracker.current_input == ""
        assert tracker.detected_words == []
        assert tracker.previous_letter is None
        assert tracker.config == config
    
    def test_add_gesture(self, tracker):
        """Test adding gestures to buffer."""
        tracker.add_gesture('A')
        tracker.add_gesture('A')
        tracker.add_gesture('B')
        
        assert len(tracker.gesture_buffer) == 3
        assert tracker.gesture_buffer == ['A', 'A', 'B']
    
    def test_buffer_size_limit(self, tracker):
        """Test buffer doesn't exceed configured size."""
        buffer_size = tracker.config.gesture.buffer_size
        
        # Add more than buffer size
        for i in range(buffer_size + 5):
            tracker.add_gesture('A')
        
        assert len(tracker.gesture_buffer) == buffer_size
    
    def test_get_most_frequent_gesture_above_threshold(self, tracker):
        """Test getting frequent gesture above threshold."""
        # Add 8 'A's and 2 'B's (80% A)
        for _ in range(8):
            tracker.add_gesture('A')
        for _ in range(2):
            tracker.add_gesture('B')
        
        # With threshold 0.7, should return 'A'
        result = tracker.get_most_frequent_gesture()
        assert result == 'A'
    
    def test_get_most_frequent_gesture_below_threshold(self, tracker):
        """Test returns None if below threshold."""
        # Add 5 'A's and 5 'B's (50% each)
        for _ in range(5):
            tracker.add_gesture('A')
        for _ in range(5):
            tracker.add_gesture('B')
        
        # With threshold 0.7, should return None
        result = tracker.get_most_frequent_gesture()
        assert result is None
    
    def test_process_gesture_adds_to_input(self, tracker):
        """Test processing gesture adds to current input."""
        result = tracker.process_gesture('H')
        
        assert result is True
        assert tracker.current_input == 'H'
        assert tracker.previous_letter == 'H'
    
    def test_process_gesture_prevents_repeats(self, tracker):
        """Test same letter not added immediately."""
        tracker.process_gesture('A')
        result = tracker.process_gesture('A')  # Same letter
        
        # Should not add second 'A' immediately
        assert tracker.current_input == 'A'  # Still just one
    
    def test_process_gesture_allows_repeat_after_delay(self, tracker):
        """Test same letter can be added after delay."""
        tracker.process_gesture('A')
        
        # Manipulate time
        tracker.last_letter_time = time.time() - 2.0  # 2 seconds ago
        
        result = tracker.process_gesture('A')
        
        # Should add second 'A' after delay
        assert result is True
        assert tracker.current_input == 'AA'
    
    def test_should_finalize_word_after_delay(self, tracker):
        """Test word finalization after inactivity."""
        tracker.current_input = "HELLO"
        tracker.last_activity_time = time.time() - 4.0  # 4 seconds ago
        
        # Should finalize (delay is 3.5 seconds)
        assert tracker.should_finalize_word() is True
    
    def test_should_not_finalize_word_too_soon(self, tracker):
        """Test word not finalized if recent activity."""
        tracker.current_input = "HELLO"
        tracker.last_activity_time = time.time() - 1.0  # 1 second ago
        
        # Should not finalize yet
        assert tracker.should_finalize_word() is False
    
    def test_finalize_word_with_autocorrect(self, tracker):
        """Test word finalization with autocorrection."""
        tracker.current_input = "HEL"
        
        # Mock word predictor
        mock_predictor = Mock()
        mock_predictor.get_suggestions = Mock(return_value=['HELLO'])
        
        result = tracker.finalize_word(mock_predictor)
        
        assert result == 'HELLO'
        assert 'HELLO' in tracker.detected_words
        assert tracker.current_input == ""  # Cleared
    
    def test_get_translation_joins_words(self, tracker):
        """Test getting complete translation."""
        tracker.detected_words = ['HELLO', 'WORLD']
        
        translation = tracker.get_translation()
        
        assert translation == 'HELLO WORLD'
