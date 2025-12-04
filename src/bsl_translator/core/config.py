"""
BSL Translator - Configuration Module
Contains all configuration constants and settings.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CameraConfig:
    """Camera capture settings."""
    device_id: int = 0
    width: int = 640
    height: int = 480
    
    
@dataclass
class MediaPipeConfig:
    """MediaPipe hand detection settings."""
    static_image_mode: bool = False
    max_num_hands: int = 2
    min_detection_confidence: float = 0.3
    

@dataclass
class GestureConfig:
    """Gesture recognition settings."""
    threshold: float = 0.7  # Confidence threshold for gesture detection
    buffer_size: int = 10  # Number of frames to buffer for stability
    letter_repeat_delay: float = 1.0  # Seconds before allowing same letter again
    word_finalize_delay: float = 3.5  # Seconds of pause before finalizing word
    

@dataclass
class ModelConfig:
    """ML model settings."""
    default_paths: list[str] = None
    expected_features: int = 84  # 42 landmarks * 2 coords
    hand_landmark_count: int = 42  # Padding for single hand
    
    def __post_init__(self):
        if self.default_paths is None:
            self.default_paths = [
                'model.p',
                'models/bsl_classifier.pkl',
                'project/server/model.p',  # Legacy location
            ]


@dataclass
class Config:
    """Main configuration combining all settings."""
    camera: CameraConfig = None
    mediapipe: MediaPipeConfig = None
    gesture: GestureConfig = None
    model: ModelConfig = None
    
    def __post_init__(self):
        if self.camera is None:
            self.camera = CameraConfig()
        if self.mediapipe is None:
            self.mediapipe = MediaPipeConfig()
        if self.gesture is None:
            self.gesture = GestureConfig()
        if self.model is None:
            self.model = ModelConfig()


# Default configuration instance
default_config = Config()
