"""
BSL Translator - Standalone Webcam Mode
Allows users to test BSL translation directly with their PC webcam without needing a Raspberry Pi or Firebase.
"""

import os
import cv2
import mediapipe as mp
import pickle
import numpy as np
from collections import Counter
import time
import nltk
from nltk.corpus import words
import difflib
import json
import sys

print("="*60)
print("BSL Translator - PC Webcam Mode")
print("="*60)
print("Press 'Q' or 'ESC' to quit")
print("="*60)

# Download NLTK word corpus
try:
    nltk.download('words', quiet=True)
    word_dict = set(word.upper() for word in words.words())
    print("✓ Word dictionary loaded successfully")
except Exception as e:
    print(f"⚠ Error downloading NLTK words: {e}")
    # Create a fallback dictionary with common words if NLTK fails
    word_dict = set(['HELLO', 'THANK', 'YOU', 'PLEASE', 'HELP', 'GOOD', 'BAD', 
                     'YES', 'NO', 'MAYBE', 'HOW', 'WHAT', 'WHERE', 'WHEN', 'WHO'])

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
print("✓ MediaPipe initialized")

# Load model
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    
    model_paths = [
        'model.p',
        os.path.join(script_dir, 'model.p'),
        os.path.join(current_dir, 'model.p'),
        os.path.join(os.path.dirname(current_dir), 'model.p'),
        os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'model.p')
    ]
    
    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"Found model at: {model_path}")
            try:
                with open(model_path, 'rb') as f:
                    model_dict = pickle.load(f)
                    if 'model' in model_dict:
                        model = model_dict['model']
                        print(f"✓ Model loaded successfully")
                        model_loaded = True
                        break
                    else:
                        print(f"⚠ Model file doesn't contain 'model' key")
            except Exception as e:
                print(f"⚠ Failed to load model from {model_path}: {e}")
                continue
    
    if not model_loaded:
        print("\n❌ ERROR: Model file not found!")
        print("Please ensure model.p exists in one of these locations:")
        for path in model_paths[:3]:
            print(f"  - {path}")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# Variables for gesture detection
gesture_list = []
current_input = ""
detected_words = []
previous_letter = None
last_time = time.time()
last_detected_time = time.time()
translation_data_file = "translation_data.json"

def update_translation_data(translation):
    """Update the translation data file used by the Express server."""
    try:
        data = {
            "translation": translation,
            "timestamp": int(time.time())
        }
        
        with open(translation_data_file, 'w') as f:
            json.dump(data, f)
            
        return True
    except Exception as e:
        print(f"Error updating translation data: {e}")
        return False

def drawLandmarks(img, hand_landmarks):
    """Draw hand landmarks on the frame."""
    mp_drawing.draw_landmarks(
        img,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )

def get_most_frequent_gesture(gesture_list, threshold=0.7):
    """Get the most frequently detected gesture if it meets the threshold."""
    counts = Counter(gesture_list)
    total_gestures = len(gesture_list)
    for gesture, count in counts.items():
        if count / total_gestures >= threshold:
            return gesture
    return None

def get_word_suggestions(fragment, dictionary, num=3):
    """Get word suggestions based on the current input fragment."""
    if not fragment:
        return []
    return difflib.get_close_matches(fragment, dictionary, n=num, cutoff=0.6)

def process_frame(frame):
    """Process a single frame using the BSL model."""
    global gesture_list, current_input, detected_words, previous_letter, last_time, last_detected_time
    
    if frame is None:
        return None, {
            "current_input": current_input,
            "suggestions": [],
            "detected_words": detected_words
        }
    
    data_aux = []
    x_coords = []
    y_coords = []
    
    try:
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        
        predicted_character = ""
        if result.multi_hand_landmarks:
            # Draw landmarks on all detected hands
            for hand_landmarks in result.multi_hand_landmarks[:2]:
                drawLandmarks(frame, hand_landmarks)
            
            # Process hand data for prediction
            if len(result.multi_hand_landmarks) == 1:
                hand = result.multi_hand_landmarks[0]
                for landmark in hand.landmark:
                    data_aux.extend([landmark.x, landmark.y])
                    x_coords.append(landmark.x)
                    y_coords.append(landmark.y)
                data_aux.extend([0.0] * 42)  # Padding for second hand
            else:
                for hand in result.multi_hand_landmarks[:2]:
                    for landmark in hand.landmark:
                        data_aux.extend([landmark.x, landmark.y])
                        x_coords.append(landmark.x)
                        y_coords.append(landmark.y)
            
            if x_coords and y_coords:
                x1 = int(min(x_coords) * W) - 10
                y1 = int(min(y_coords) * H) - 10
                x2 = int(max(x_coords) * W) + 10
                y2 = int(max(y_coords) * H) + 10
                
                try:
                    data_aux_np = np.array(data_aux).reshape(1, -1)
                    
                    # Check if the model input shape matches
                    expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                    actual_features = data_aux_np.shape[1]
                    
                    if expected_features is not None and expected_features != actual_features:
                        # Try to fix by padding or truncating
                        if actual_features < expected_features:
                            padding = np.zeros((1, expected_features - actual_features))
                            data_aux_np = np.hstack((data_aux_np, padding))
                        else:
                            data_aux_np = data_aux_np[:, :expected_features]
                    
                    prediction = model.predict(data_aux_np)
                    predicted_character = prediction[0].upper()
                    
                    gesture_list.append(predicted_character)
                    if len(gesture_list) > 10:
                        gesture_list.pop(0)
                    
                    most_frequent_gesture = get_most_frequent_gesture(gesture_list)
                    if most_frequent_gesture:
                        current_time = time.time()
                        if most_frequent_gesture != previous_letter or (current_time - last_time) > 1:
                            current_input += most_frequent_gesture.upper()
                            previous_letter = most_frequent_gesture
                            last_time = current_time
                            last_detected_time = current_time
                except Exception as e:
                    print(f"❌ Error in prediction: {e}")
        
        suggestions = get_word_suggestions(current_input, word_dict)
        
        # Finalize word after 3.5 seconds of no new input
        if time.time() - last_detected_time > 3.5 and current_input:
            finalized_word = suggestions[0] if suggestions else current_input
            detected_words.append(finalized_word)
            current_input = ""
            last_detected_time = time.time()
            
            # Create final sentence and update outputs
            final_sentence = " ".join(detected_words)
            update_translation_data(final_sentence)
        
        # Return the processed frame and current translation
        return frame, {
            "current_input": current_input,
            "suggestions": suggestions,
            "detected_words": detected_words
        }
    except Exception as e:
        print(f"❌ Exception in process_frame: {e}")
        return frame, {
            "current_input": current_input,
            "suggestions": get_word_suggestions(current_input, word_dict),
            "detected_words": detected_words
        }

def display_debug_info(frame, translation_info):
    """Add debug info overlay to the frame."""
    try:
        H, W, _ = frame.shape
        
        # Create a semi-transparent overlay for better readability
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, H - 150), (W, H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Add current input buffer
        cv2.putText(frame, f"Input: {translation_info['current_input']}", 
                    (10, H - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add word suggestions
        if translation_info['suggestions']:
            suggestions_str = ", ".join(translation_info['suggestions'][:3])
            cv2.putText(frame, f"Suggestions: {suggestions_str}", 
                        (10, H - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Add detected words
        detected_str = " ".join(translation_info['detected_words'][-5:])
        cv2.putText(frame, f"Translation: {detected_str}", 
                    (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add instructions at the top
        cv2.putText(frame, "Press 'Q' or 'ESC' to quit", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    except Exception as e:
        print(f"❌ Error in display_debug_info: {e}")
        return frame

def main():
    """Main function to run the webcam BSL translator."""
    print("\n" + "="*60)
    print("Starting PC Webcam Mode...")
    print("="*60 + "\n")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam")
        print("Please check:")
        print("  - Your webcam is connected")
        print("  - No other application is using the webcam")
        print("  - You have granted camera permissions")
        sys.exit(1)
    
    print("✓ Webcam opened successfully")
    
    # Set webcam properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize translation data
    update_translation_data("BSL Translation will appear here...")
    
    # Create window
    cv2.namedWindow("BSL Translator - PC Webcam Mode", cv2.WINDOW_NORMAL)
    
    print("\n✨ BSL Translator is ready!")
    print("📹 Show BSL hand gestures to the camera")
    print("🔤 Letters will be detected and translated to words")
    print("\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Failed to read frame from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame, translation_info = process_frame(frame)
            
            # Display debug info on frame
            if processed_frame is not None:
                display_frame = display_debug_info(processed_frame, translation_info)
                cv2.imshow("BSL Translator - PC Webcam Mode", display_frame)
            
            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or 'q'
                print("\n👋 Exiting...")
                break
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam released")
        print("✓ Windows closed")
        print("\n" + "="*60)
        print("Thank you for using BSL Translator!")
        print("="*60)

if __name__ == "__main__":
    main()
