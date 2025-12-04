"""
BSL Translator - Firebase Stream Client
Handles video stream upload and retrieval from Firebase Realtime Database.

This module provides utilities for:
- Creating test streams using a webcam
- Viewing existing streams
- Listing active streams
"""

import os
import time
import json
import base64
import argparse
import logging
from typing import Optional, Dict, List, Any

import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

logger = logging.getLogger(__name__)

# Environment variable keys
FIREBASE_URL_ENV = "FIREBASE_DATABASE_URL"
FIREBASE_CREDS_ENV = "FIREBASE_CREDENTIALS_PATH"

# Default Firebase URL (can be overridden by env var)
DEFAULT_FIREBASE_URL = "https://bsltranslator-93f00-default-rtdb.europe-west1.firebasedatabase.app/"


def base64_to_image(base64_string: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to an OpenCV image.
    
    Args:
        base64_string: Base64 encoded image data
        
    Returns:
        OpenCV image array or None if decoding fails
    """
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert OpenCV image to base64 string.
    
    Args:
        image: OpenCV BGR image
        
    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def create_test_stream() -> Optional[str]:
    """
    Create a test stream in Firebase using a webcam.
    
    Returns:
        Stream ID if successful, None if webcam failed
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.error("Could not open webcam")
        return None
    
    stream_id = f"test_stream_{int(time.time())}"
    stream_ref = db.reference(f'/streams/{stream_id}')
    
    # Set initial stream info
    stream_ref.set({
        'status': 'active',
        'started_at': int(time.time() * 1000),
        'resolution': '640x480',
        'frame_rate': 15,
        'latest_frame': {
            'frame_number': 0,
            'timestamp': int(time.time() * 1000),
            'data': ''
        }
    })
    
    logger.info(f"Created test stream: {stream_id}")
    
    try:
        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from webcam")
                break
                
            # Resize frame to reduce data size
            frame = cv2.resize(frame, (640, 480))
            
            # Convert to base64
            base64_data = image_to_base64(frame)
            
            # Update the latest frame
            latest_frame_ref = stream_ref.child('latest_frame')
            latest_frame_ref.update({
                'frame_number': frame_number,
                'timestamp': int(time.time() * 1000),
                'data': base64_data
            })
            
            # Display the frame locally
            cv2.imshow('Test Stream', frame)
            
            frame_number += 1
            time.sleep(0.066)  # ~15 FPS
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    
    finally:
        stream_ref.update({'status': 'ended'})
        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Test stream {stream_id} ended")
    
    return stream_id


def get_stream_frames(stream_id: str) -> None:
    """
    Get frames from an existing stream and display them.
    
    Args:
        stream_id: Firebase stream ID
    """
    stream_ref = db.reference(f'/streams/{stream_id}')
    stream_info = stream_ref.get()
    
    if not stream_info:
        logger.error(f"Stream {stream_id} not found")
        return
    
    logger.info(f"Connected to stream: {stream_id}")
    logger.info(f"Status: {stream_info.get('status', 'unknown')}")
    
    if 'started_at' in stream_info:
        started_at = time.strftime(
            '%Y-%m-%d %H:%M:%S', 
            time.localtime(stream_info['started_at'] / 1000)
        )
        logger.info(f"Started at: {started_at}")
    
    latest_frame_ref = stream_ref.child('latest_frame')
    last_frame_number = -1
    
    try:
        cv2.namedWindow('Stream Viewer', cv2.WINDOW_NORMAL)
        
        while True:
            frame_data = latest_frame_ref.get()
            
            if not frame_data or 'data' not in frame_data:
                logger.debug("No frame data available, waiting...")
                time.sleep(0.5)
                continue
            
            # Check if this is a new frame
            current_frame_num = frame_data.get('frame_number', -1)
            if current_frame_num == last_frame_number:
                time.sleep(0.1)
                continue
            
            # Convert base64 to image
            frame = base64_to_image(frame_data['data'])
            if frame is None:
                logger.warning("Failed to decode frame")
                time.sleep(0.5)
                continue
            
            # Display the frame
            cv2.imshow('Stream Viewer', frame)
            last_frame_number = current_frame_num
            
            # Save frame data to file for BSL interpreter
            frame_info = {
                'frame_number': frame_data.get('frame_number', 0),
                'timestamp': frame_data.get('timestamp', 0),
                'data': frame_data.get('data', '')
            }
            
            with open('current_frame.json', 'w') as f:
                json.dump(frame_info, f)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        logger.info("Viewer interrupted by user")
    
    finally:
        cv2.destroyAllWindows()
        logger.info(f"Disconnected from stream {stream_id}")


def list_active_streams() -> List[str]:
    """
    List all active streams in the database.
    
    Returns:
        List of active stream IDs
    """
    streams_ref = db.reference('/streams')
    streams = streams_ref.get()
    
    if not streams:
        logger.info("No streams found in the database")
        return []
    
    active_streams: List[str] = []
    
    for stream_id, stream_data in streams.items():
        status = stream_data.get('status', 'unknown')
        started_at = stream_data.get('started_at', 0)
        
        if started_at:
            started_str = time.strftime(
                '%Y-%m-%d %H:%M:%S', 
                time.localtime(started_at / 1000)
            )
        else:
            started_str = 'Unknown'
        
        logger.info(f"Stream: {stream_id} | Status: {status} | Started: {started_str}")
        
        if status == 'active':
            active_streams.append(stream_id)
    
    logger.info(f"Total streams: {len(streams)} | Active: {len(active_streams)}")
    return active_streams


def get_firebase_url() -> str:
    """Get Firebase database URL from environment or use default."""
    return os.getenv(FIREBASE_URL_ENV, DEFAULT_FIREBASE_URL)


def get_credentials_path() -> Optional[str]:
    """
    Get Firebase credentials path from environment or search common locations.
    
    Returns:
        Path to credentials file or None if not found
    """
    # Check environment variable first
    env_path = os.getenv(FIREBASE_CREDS_ENV)
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Search common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(script_dir, 'firebase-credentials.json'),
        os.path.join(script_dir, '..', 'firebase-credentials.json'),
        'firebase-credentials.json',
        'bsltranslator-93f00-firebase-adminsdk-fbsvc-55978db132.json',
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    return None


def initialize_firebase() -> bool:
    """
    Initialize Firebase Admin SDK.
    
    Returns:
        True if successful, False otherwise
    """
    cred_path = get_credentials_path()
    
    if not cred_path:
        logger.error("Firebase credentials file not found")
        logger.error(f"Set {FIREBASE_CREDS_ENV} environment variable or place credentials file")
        return False
    
    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': get_firebase_url()
        })
        logger.info(f"Firebase initialized with credentials from: {cred_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        return False


def main() -> None:
    """Command-line interface for Firebase stream operations."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Firebase Stream Handler')
    parser.add_argument('--create', action='store_true', 
                        help='Create a test stream using webcam')
    parser.add_argument('--list', action='store_true', 
                        help='List all streams')
    parser.add_argument('--view', metavar='STREAM_ID', 
                        help='View a specific stream')
    
    args = parser.parse_args()
    
    # Initialize Firebase
    if not initialize_firebase():
        return
    
    if args.create:
        create_test_stream()
    elif args.list:
        list_active_streams()
    elif args.view:
        get_stream_frames(args.view)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()