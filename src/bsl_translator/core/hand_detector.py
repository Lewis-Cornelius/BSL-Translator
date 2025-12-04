"""
BSL Translator - Hand Detector Module
MediaPipe-based hand landmark detection wrapper.
"""
import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List

from .config import MediaPipeConfig


class HandDetector:
    """Wrapper for MediaPipe hand detection."""
    
    def __init__(self, config: Optional[MediaPipeConfig] = None):
        """
        Initialize hand detector.
        
        Args:
            config: MediaPipe configuration. Uses defaults if None.
        """
        if config is None:
            config = MediaPipeConfig()
            
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=config.static_image_mode,
            max_num_hands=config.max_num_hands,
            min_detection_confidence=config.min_detection_confidence
        )
        
    def detect(self, frame: np.ndarray) -> Tuple[Optional[List], List[Tuple[int, int, int, int]]]:
        """
        Detect hands in frame and extract landmarks.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            Tuple of (landmarks_list, bounding_boxes)
            - landmarks_list: List of hand landmarks (or None if no hands)
            - bounding_boxes: List of (x1, y1, x2, y2) for each hand
        """
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if not results.multi_hand_landmarks:
            return None, []
            
        bboxes = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            x1 = int(min(x_coords) * W) - 10
            y1 = int(min(y_coords) * H) - 10
            x2 = int(max(x_coords) * W) + 10
            y2 = int(max(y_coords) * H) + 10
            
            bboxes.append((x1, y1, x2, y2))
            
        return results.multi_hand_landmarks, bboxes
    
    def draw_landmarks(self, frame: np.ndarray, hand_landmarks) -> None:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: Image to draw on (modified in-place)
            hand_landmarks: MediaPipe hand landmarks
        """
        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
    
    def extract_features(self, landmarks_list, num_hands: int = 2) -> Optional[np.ndarray]:
        """
        Extract feature vector from hand landmarks.
        
        Args:
            landmarks_list: List of hand landmarks from MediaPipe
            num_hands: Expected number of hands (1 or 2)
            
        Returns:
            Feature array of shape (1, 84) or None if invalid
        """
        if not landmarks_list:
            return None
            
        data_aux = []
        
        # Extract coordinates from detected hands
        for hand in landmarks_list[:num_hands]:
            for landmark in hand.landmark:
                data_aux.extend([landmark.x, landmark.y])
        
        # Pad with zeros if only one hand detected
        if len(landmarks_list) == 1 and num_hands == 2:
            data_aux.extend([0.0] * 42)  # 21 landmarks * 2 coords
            
        return np.array(data_aux).reshape(1, -1)
    
    def release(self):
        """Clean up resources."""
        self.hands.close()
