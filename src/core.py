import numpy as np

from typing import List, Tuple, Optional, Dict, Any

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from memory_manager import MemoryManager

class PersonalObjectRecognizer:
    """Main application class for personal object recognition."""
    def __init__(self):
        self.config = load_config()
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.memory = MemoryManager()

        # Load current state
        self._prototypes, self._labels = self.memory.get_all_prototypes()

    def register_object(self, object_name: str, images: List[np.ndarray]) -> bool:
        """
        Registers a new object from a list of images.
        
        Args:
            object_name: Name for the object.
            images: List of 4 BGR images (differents views).
            
        Returns:
            True if registration successful.
        """
        if len(images) != 4:
            print(f"Warning: Expected 4 images, got {len(images)}.")
        
        # Extract embeddings
        embeddings = self.extractor.extract_batch(images)

        # Store in memory
        self.memory.add_object(object_name, embeddings)

        # Update cache
        self._prototypes, self._labels = self.memory.get_all_prototypes()

        return True
    
    def recognize_frame(self, frame: np.ndarray) \
        -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], Dict[str, Any]]:
        """
        Recognizes objects in a frame.
        
        Returns:
            (label, bounding_box, debug_info)
        """
        # Detect object
        box = self.detector.detect(frame)
        if box is None:
            return None, None, {'status': 'no_object'}
        
        # Crop and extract features
        crop = self.detector.crop_object(frame, box)
        embedding = self.extractor.extract(crop)

        # Compare with all prototypes
        if len(self._prototypes) == 0:
            return None, box, {'status': 'no_objects_in_db'}

        similarities = np.dot(self._prototypes, embedding)
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_label = self._labels[best_idx]
        
        # Threshold
        if best_similarity >= self.config['recognition']['similarity_threshold_high']:
            return best_label, box, {
                'status': 'recognized',
                'similarity': float(best_similarity),
                'all_similarities': [float(s) for s in similarities]
            }
        else:
            return None, box, {
                'status': 'unknown',
                'similarity': float(best_similarity)
            }
        
    def clear_database(self):
        """Clears all registered objects."""
        self.memory.clear()
        self._prototypes, self._labels = [], []
        