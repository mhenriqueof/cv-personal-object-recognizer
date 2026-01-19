import cv2
import numpy as np

from typing import List

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
def preprocess_image_to_bgr(image: np.ndarray) -> np.ndarray:
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
        
        # Detect object
        box = detector.detect(cv_image)
        
        if box is None:
            return "No object detected in image", None
        
        # Crop and extract features
        crop = detector.crop_object(cv_image, box)
        embedding = extractor.extract(crop)
        
        # Compare with database
        prototypes, labels = memory.get_all_prototypes()

        if len(prototypes) == 0:
            return "No objects in database. Please register objects first.", None

        # Calculate similarities
        prototypes_arrays = np.array(prototypes)
        similarities = np.dot(prototypes_arrays, embedding)

        # Get best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_label = labels[best_idx]

        # Thresholds
        config = load_config()
        threshold_high = config['recognition']['similarity_threshold_high']
        
        if best_similarity => threshold_high:
            result = f" {best_label} (confidence: {best_similarity:.2f})"
        else:
            result = f" Unknown object (closest: {best_label}, similarity: {best_similarity:.2f})"
        
        # Draw bounding box on image
        x1, y1, x2, y2 = box
        result_image = cv_image.copy()
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        return result, result_image

    except Exception as e:
        return f" Error: {str(e)}", None
    
def get_database_status():
    """Gets current database status."""
    prototypes, labels = memory.get_all_prototypes()

    if len(labels) == 0:
        return "Database is empty. Please register objects first."
    
    status = f" **Database Status:** {len(labels)} object(s) registered \
               \n\n"
    for i, label in enumerate(labels, 1):
        status += f"{i}. **{label}** \
                    \n"

    return status

def clear_database():
    """Clears all objects from database."""
    memory.clear()
    return " Database cleared successfully!"

# Create Gradio interface