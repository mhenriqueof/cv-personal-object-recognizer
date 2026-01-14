import cv2

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

def live_recognition():
    """
    Real-time object recognition from webcam.
    """
    config = load_config()
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    memory = MemoryManager()

    # Get recognition thresholds
    threshold_high = config['recognition']['similarity_threshold_high']
    threshold_low = config['recognition']['similarity_threshold_low']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: could not open webcam.")
        return
    