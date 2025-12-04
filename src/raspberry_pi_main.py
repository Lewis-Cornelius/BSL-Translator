"""
BSL Translator - Raspberry Pi / Firebase Main
Processes BSL video streams from Firebase Realtime Database.

Usage:
    python src/raspberry_pi_main.py [stream_id]
    
Environment Variables:
    FIREBASE_DATABASE_URL: Firebase Realtime Database URL
    FIREBASE_CREDENTIALS_PATH: Path to Firebase credentials JSON
    
If no stream_id provided, will prompt for stream selection.
"""

import os
import sys
import time
import json
import base64
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import requests

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


# Import gesture tracker from demo (could be moved to core module later)
from demo_standalone import GestureTracker


class FirebaseStreamProcessor:
    """Processes BSL video streams from Firebase."""
    
    def __init__(self, stream_id: str) -> None:
        """Initialize Firebase stream processor."""
        self.stream_id = stream_id
        self.config = Config()
        
        # Initialize components
        logger.info("Initializing components...")
        self.hand_detector = HandDetector(self.config.mediapipe)
        self.gesture_classifier = GestureClassifier(config=self.config.model)
        self.word_predictor = WordPredictor()
        self.tracker = GestureTracker(self.config)
        
        # Firebase references
        self.stream_ref = db.reference(f'/streams/{stream_id}')
        self.frame_ref = db.reference(f'/streams/{stream_id}/latest_frame')
        
        # Output file
        self.translation_file = Path("translation_data.json")
        
        # Display
        self.debug_window = True
        
    def check_prerequisites(self) -> bool:
        """Verify model and stream."""
        if not self.gesture_classifier.is_loaded():
            logger.error("Model not loaded")
            return False
        
        # Check stream exists
        stream_data = self.stream_ref.get()
        if not stream_data:
            logger.error(f"Stream {self.stream_id} not found")
            return False
        
        logger.info(f"Stream {self.stream_id} verified")
        if 'piid' in stream_data:
            logger.info(f"Stream PIID: {stream_data['piid']}")
        
        return True
    
    def base64_to_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Convert base64 to OpenCV image."""
        try:
            img_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None
    
    def update_translation_file(self, text: str) -> None:
        """Update translation file and HTTP endpoint."""
        try:
            data = {
                "translation": text,
                "timestamp": int(time.time())
            }
            self.translation_file.write_text(json.dumps(data, indent=2))
            
            # Try HTTP update
            try:
                requests.post(
                    'http://localhost:3000/update-translation',
                    json={"translation": text},
                    timeout=1
                )
            except Exception:
                pass  # File update is primary
            
            logger.info(f"Translation: {text}")
        except Exception as e:
            logger.warning(f"Could not update translation: {e}")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process frame - same logic as demo_standalone."""
        # Detect hands
        landmarks, bboxes = self.hand_detector.detect(frame)
        
        translation_info: Dict[str, Any] = {
            "current_input": self.tracker.current_input,
            "suggestions": [],
            "detected_words": self.tracker.detected_words
        }
        
        if landmarks:
            # Draw landmarks
            for hand_landmarks in landmarks:
                self.hand_detector.draw_landmarks(frame, hand_landmarks)
            
            # Classify
            features = self.hand_detector.extract_features(landmarks, num_hands=2)
            if features is not None:
                predicted_letter = self.gesture_classifier.predict(features)
                
                if predicted_letter:
                    self.tracker.add_gesture(predicted_letter)
                    stable_gesture = self.tracker.get_most_frequent_gesture()
                    
                    if stable_gesture:
                        if self.tracker.process_gesture(stable_gesture):
                            logger.debug(f"Letter: {stable_gesture}")
        
        # Get suggestions
        if self.tracker.current_input:
            translation_info["suggestions"] = self.word_predictor.get_suggestions(
                self.tracker.current_input, num=3
            )
        
        # Finalize word
        if self.tracker.should_finalize_word():
            self.tracker.finalize_word(self.word_predictor)
            translation = self.tracker.get_translation()
            self.update_translation_file(translation)
        
        translation_info["current_input"] = self.tracker.current_input
        translation_info["detected_words"] = self.tracker.detected_words
        
        return frame, translation_info
    
    def draw_ui(self, frame: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """Draw UI overlay."""
        H, W, _ = frame.shape
        
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
                (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # Translation
        detected_str = " ".join(info['detected_words'][-3:])
        cv2.putText(
            frame, f"Detected: {detected_str}",
            (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        return frame
    
    def run(self) -> None:
        """Main processing loop."""
        if not self.check_prerequisites():
            logger.error("Pre-flight checks failed")
            sys.exit(1)
        
        # Create debug window
        window_name = "BSL Processing"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            logger.info("Debug window created")
        except Exception as e:
            logger.warning(f"No debug window: {e}")
            self.debug_window = False
        
        # Initialize
        self.update_translation_file("BSL Translation will appear here...")
        
        last_frame_number = -1
        last_status_check = time.time()
        error_count = 0
        max_errors = 10
        
        logger.info("=" * 60)
        logger.info(f"Processing stream: {self.stream_id}")
        logger.info("Press 'Q' or ESC to quit")
        logger.info("=" * 60)
        
        try:
            while True:
                # Check stream status periodically
                if time.time() - last_status_check > 30:
                    stream_data = self.stream_ref.get()
                    if not stream_data or stream_data.get('status') != 'active':
                        logger.info("Stream no longer active")
                        break
                    last_status_check = time.time()
                
                # Get frame
                frame_data = self.frame_ref.get()
                
                if not frame_data or 'data' not in frame_data:
                    time.sleep(0.5)
                    continue
                
                # Skip old frames
                if frame_data.get('frame_number') == last_frame_number:
                    time.sleep(0.1)
                    continue
                
                # Decode frame
                frame = self.base64_to_image(frame_data['data'])
                if frame is None:
                    error_count += 1
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}). Restarting...")
                        time.sleep(5)
                        error_count = 0
                    continue
                
                # Process
                processed_frame, translation_info = self.process_frame(frame)
                error_count = 0
                
                # Display
                if processed_frame is not None and self.debug_window:
                    try:
                        display_frame = self.draw_ui(processed_frame.copy(), translation_info)
                        cv2.imshow(window_name, display_frame)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27 or key == ord('q'):
                            break
                    except Exception as e:
                        logger.error(f"Display error: {e}")
                
                last_frame_number = frame_data.get('frame_number', -1)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            if self.debug_window:
                cv2.destroyAllWindows()
            self.hand_detector.release()
            
            logger.info("=" * 60)
            logger.info("Stream processing ended")
            logger.info("=" * 60)


def initialize_firebase() -> bool:
    """
    Initialize Firebase Admin SDK.
    
    Supports environment variables:
    - FIREBASE_DATABASE_URL: Firebase Realtime Database URL
    - FIREBASE_CREDENTIALS_PATH: Path to credentials JSON file
    """
    # Get database URL from environment or use default
    database_url = os.getenv(
        'FIREBASE_DATABASE_URL',
        'https://bsltranslator-93f00-default-rtdb.europe-west1.firebasedatabase.app/'
    )
    
    env_cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH')
    
    try:
        script_dir = Path(__file__).parent
        cwd = Path.cwd()
        
        credential_paths = [
            env_cred_path,
            'bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json',
            script_dir / 'bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json',
            cwd / 'bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json',
            'firebase-credentials.json',
        ]
        
        for cred_path in credential_paths:
            if cred_path and Path(cred_path).exists():
                cred = credentials.Certificate(str(cred_path))
                firebase_admin.initialize_app(cred, {
                    'databaseURL': database_url
                })
                logger.info(f"Firebase initialized: {Path(cred_path).name}")
                return True
        
        logger.error("Firebase credentials not found!")
        logger.error("Set FIREBASE_CREDENTIALS_PATH environment variable")
        return False
        
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
        return False


def get_stream_id() -> Optional[str]:
    """Get stream ID from command line or file."""
    if len(sys.argv) > 1:
        logger.info(f"Using stream ID from command line: {sys.argv[1]}")
        return sys.argv[1]
    
    # Try reading from file
    try:
        stream_id = Path('active_stream.txt').read_text().strip()
        logger.info(f"Using stream ID from file: {stream_id}")
        return stream_id
    except FileNotFoundError:
        logger.error("No stream ID provided and active_stream.txt not found")
        logger.error("Usage: python raspberry_pi_main.py <stream_id>")
        return None
    except Exception as e:
        logger.error(f"Error reading stream ID file: {e}")
        return None


def main() -> None:
    """Entry point for Raspberry Pi mode."""
    logger.info("=" * 60)
    logger.info("BSL Bridge Integration Starting...")
    logger.info("=" * 60)
    
    # Initialize Firebase
    if not initialize_firebase():
        sys.exit(1)
    
    # Get stream ID
    stream_id = get_stream_id()
    if not stream_id:
        sys.exit(1)
    
    # Process stream
    processor = FirebaseStreamProcessor(stream_id)
    processor.run()


if __name__ == "__main__":
    main()