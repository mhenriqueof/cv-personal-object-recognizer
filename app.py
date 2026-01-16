import cv2
import numpy as np

from typing import Optional

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

# Initialize components (cached)
def init_components():
    """Initializes model components once."""
    config = load_config()
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    memory = MemoryManager()
    return config, detector, extractor, memory

config, detector, extractor, memory = init_components()

# Helper functions
def preprocess_image_to_bgr(image) -> Optional[np.ndarray]:
    """Converts Gradio image to OpenCV format."""
    if image is None:
        return None
    # Gradio image is numpy array in RGB format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image_bgr

def register_object(images, label) -> str:
    """Registers new object from 4 uploaded iamges."""
    if not label or len(images) != 4:
        return " Please provide a label and exactly 4 images."
    
    try:
        # Convert Gradio images to OpenCV format
        cv_images = []
        for img in images:
            if img is None:
                return " One or more images are invalid."
            cv_img = preprocess_image_to_bgr(img)
            cv_images.append(cv_img)
            
        # Extract embeddings
        embeddings = extractor.extract_batch(cv_images)

        # Add to memory
        memory.add_object(label, embeddings)

        # Get updated database
        prototypes, labels_list = memory.get_all_prototypes()

        return  f" Successfully registered '{label}'. Database now has {len(labels_list)} objects."
    
    except Exception as e:
        return f" Error: {str(e)}"

def recognize_image(image):
    """Recognizes object in uploaded image."""
    if image is None:
        return " Please upload an image."
    
    try:
        # Convert image
        cv_image = preprocess_image_to_bgr(image)
