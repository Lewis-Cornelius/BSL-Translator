"""
BSL Translator - Standalone Webcam Demo
Test BSL translation directly with your PC webcam - no Raspberry Pi or Firebase needed.

Usage:
    python src/demo_standalone.py
    
Controls:
    Q or ESC - Quit
"""

import sys
import time
import json
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

# Import our core modules
from bsl_translator.core import (
    Config,
    HandDetector,
    GestureClassifier,
    WordPredictor,
)
from bsl_translator.utils import setup_logging, get_logger

# Setup logging
logger = setup_logging(level="INFO")
logger = get_logger(__name__)


class GestureTracker:
    """Tracks gestures over time and manages word detection."""
    
    def __init__(self, config: Config):
        """Initialize gesture tracker with configuration."""
        self.config = config
        self.gesture_buffer = []
        self.current_input = ""
        self.detected_words = []
        self.previous_letter = None
        self.last_letter_time = time.time()
        self.last_activity_time = time.time()
        
    def add_gesture(self, gesture: str) -> None:
        """Add a detected gesture to the buffer."""
        self.gesture_buffer.append(gesture)
        
        # Keep buffer at configured size
        if len(self.gesture_buffer) > self.config.gesture.buffer_size:
            self.gesture_buffer.pop(0)
    
    def get_most_frequent_gesture(self) -> Optional[str]:
        """Get most frequent gesture if it meets threshold."""
        if not self.gesture_buffer:
            return None
            
        counts = Counter(self.gesture_buffer)
        total = len(self.gesture_buffer)
        
        for gesture, count in counts.items():
            if count / total >= self.config.gesture.threshold:
                return gesture
        
        return None
    
    def process_gesture(self, gesture: str) -> bool:
        """
        Process a detected gesture and add to current input if valid.
        
        Returns:
            True if letter was added to input
        """
        current_time = time.time()
        
        # Check if it's a new letter or enough time has passed
        if (gesture != self.previous_letter or 
            (current_time - self.last_letter_time) > self.config.gesture.letter_repeat_delay):
            
            self.current_input += gesture.upper()
            self.previous_letter = gesture
            self.last_letter_time = current_time
            self.last_activity_time = current_time
            
            logger.debug(f"Added letter: {gesture}")
            return True
        
        return False
    
    def should_finalize_word(self) -> bool:
        """Check if we should finalize the current word."""
        if not self.current_input:
            return False
            
        time_since_activity = time.time() - self.last_activity_time
        return time_since_activity > self.config.gesture.word_finalize_delay
    
    def finalize_word(self, word_predictor: WordPredictor) -> str:
        """Finalize current input as a word with autocorrection."""
        if not self.current_input:
            return ""
        
        # Get suggestions and use best match
        suggestions = word_predictor.get_suggestions(self.current_input, num=1)
        finalized = suggestions[0] if suggestions else self.current_input
        
        self.detected_words.append(finalized)
        logger.info(f"Word finalized: {self.current_input} → {finalized}")
        
        self.current_input = ""
        self.last_activity_time = time.time()
        
        return finalized
    
    def get_translation(self) -> str:
        """Get the complete translation sentence."""
        return " ".join(self.detected_words)


class BSLDemo:
    """Main demo application for BSL translation."""
    
    def __init__(self):
        """Initialize demo components."""
        logger.info("=" * 60)
        logger.info("BSL Translator - PC Webcam Mode")
        logger.info("=" * 60)
        
        # Load configuration
        self.config = Config()
        
        # Initialize components
        logger.info("Initializing components...")
        self.hand_detector = HandDetector(self.config.mediapipe)
        self.gesture_classifier = GestureClassifier(config=self.config.model)
        self.word_predictor = WordPredictor()
        self.tracker = GestureTracker(self.config)
        
        logger.info("✓ All components initialized")
        
        # Camera
        self.cap = None
        
        # Output file for web server integration
        self.translation_file = Path("translation_data.json")
    
    def check_prerequisites(self) -> bool:
        """Run pre-flight checks."""
        logger.info("Running pre-flight checks...")
        
        # Check model
        if not self.gesture_classifier.is_loaded():
            logger.error("Model file not found!")
            logger.error("Please ensure model.p exists in project root or models/")
            return False
        
        logger.info("✓ Model loaded")
        
        # Check camera
        test_cap = cv2.VideoCapture(self.config.camera.device_id)
        if not test_cap.isOpened():
            logger.error("Could not open webcam")
            logger.error("Please check:")
            logger.error("  - Webcam is connected")
            logger.error("  - No other app is using it")
            logger.error("  - You have camera permissions")
            test_cap.release()
            return False
        
        test_cap.release()
        logger.info("✓ Webcam accessible")
        
        return True
    
    def update_translation_file(self, text: str) -> None:
        """Update translation file for web server."""
        try:
            data = {
                "translation": text,
                "timestamp": int(time.time())
            }
            self.translation_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not update translation file: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for BSL detection.
        
        Returns:
            Tuple of (processed_frame, translation_info)
        """
        # Detect hands
        landmarks, bboxes = self.hand_detector.detect(frame)
        
        translation_info = {
            "current_input": self.tracker.current_input,
            "suggestions": [],
            "detected_words": self.tracker.detected_words
        }
        
        if landmarks:
            # Draw landmarks
            for hand_landmarks in landmarks:
                self.hand_detector.draw_landmarks(frame, hand_landmarks)
            
            # Extract features and classify
            features = self.hand_detector.extract_features(landmarks, num_hands=2)
            if features is not None:
                predicted_letter = self.gesture_classifier.predict(features)
                
                if predicted_letter:
                    # Add to tracking buffer
                    self.tracker.add_gesture(predicted_letter)
                    
                    # Check if we have a stable gesture
                    stable_gesture = self.tracker.get_most_frequent_gesture()
                    if stable_gesture:
                        self.tracker.process_gesture(stable_gesture)
        
        # Get word suggestions for current input
        if self.tracker.current_input:
            translation_info["suggestions"] = self.word_predictor.get_suggestions(
                self.tracker.current_input, num=3
            )
        
        # Check if word should be finalized
        if self.tracker.should_finalize_word():
            self.tracker.finalize_word(self.word_predictor)
            
            # Update translation file
            translation = self.tracker.get_translation()
            self.update_translation_file(translation)
        
        # Update translation info
        translation_info["current_input"] = self.tracker.current_input
        translation_info["detected_words"] = self.tracker.detected_words
        
        return frame, translation_info
    
    def draw_ui(self, frame: np.ndarray, info: Dict) -> np.ndarray:
        """Draw UI overlay on frame."""
        H, W, _ = frame.shape
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, H - 150), (W, H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Current input
        cv2.putText(
            frame, f"Input: {info['current_input']}", 
            (10, H - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # Suggestions
        if info['suggestions']:
            suggestions_str = ", ".join(info['suggestions'][:3])
            cv2.putText(
                frame, f"Suggestions: {suggestions_str}",
                (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
        
        # Translation
        detected_str = " ".join(info['detected_words'][-5:])
        cv2.putText(
            frame, f"Translation: {detected_str}",
            (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # Instructions
        cv2.putText(
            frame, "Press Q or ESC to quit",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        
        return frame
    
    def run(self) -> None:
        """Main demo loop."""
        # Pre-flight checks
        if not self.check_prerequisites():
            logger.error("Pre-flight checks failed. Exiting.")
            sys.exit(1)
        
        # Open camera
        logger.info("Opening webcam...")
        self.cap = cv2.VideoCapture(self.config.camera.device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        
        # Initialize translation
        self.update_translation_file("BSL Translation will appear here...")
        
        # Create window
        window_name = "BSL Translator - PC Webcam Mode"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✨ BSL Translator is ready!")
        logger.info("📹 Show BSL hand gestures to the camera")
        logger.info("🔤 Letters will be detected and translated to words")
        logger.info("=" * 60)
        logger.info("")
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    break
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, translation_info = self.process_frame(frame)
                
                # Draw UI
                display_frame = self.draw_ui(processed_frame, translation_info)
                cv2.imshow(window_name, display_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q') or key == ord('Q'):
                    logger.info("Exiting...")
                    break
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.hand_detector.release()
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("Thank you for using BSL Translator!")
            logger.info("=" * 60)


def main():
    """Entry point for standalone demo."""
    demo = BSLDemo()
    demo.run()


if __name__ == "__main__":
    main()
