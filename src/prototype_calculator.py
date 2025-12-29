import numpy as np
from typing import Tuple, Dict, Any

from utils.setups import setup_logger
from utils.loads import load_config

class PrototypeCalculator:
    """Calculates a single prototype vector and distance threshold from a set of embeddings."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.distance_multiplayer = self.config.['registration']['distance_threshold_multiplier']
        self.logger.info(f"Prototype calculator initialized (multiplier: {self.distance_multiplayer}).")
        
    def calculate_prototype(self, embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Calculates a single prototype (mean) and a distance threshold.
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim), L2-normalized.
            
        Returns:
            prototype: Mean embedding vector (L2-normalized).
            threshold: Distance threshold for recognition (max allowed cosine distance).
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot calculate prototype from empty embeddings.")

        # 1. Calculate the mean vector (prototype)
        prototype = np.mean(embeddings, axis=0)
        # Re-normalize the prototype
        prototype = prototype / (np.linalg.norm(prototype) + 1e-9)

        # 2. Calculate cosine distances of each sample to the prototype
        # For normalized vectors, cosine_distance = 1 - cosine_similarity
        distances = 1 - np.dot(embeddings, prototype)
        
        # 3. Calculate statistics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        # 4. Define threshold: mean + k*std
        threshold = mean_distance + (self.distance_multiplayer * std_distance)

        self.logger.debug(f"Prototype stats: n={len(embeddings)}, mean_dist={mean_distance:.4f}, "
                          f"std_dist={std_distance:.4f}, threshold={threshold:.4f}")

        return prototype, threshold
    
    def batch_calculate(self, embeddings_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Calculates protypes for multiple objects.
        
        Args:
            embedding_dict: {object_label: embeddings_array}

        Returns:
            Dictionary with prototype data for each object.
        """
        results = {}

        for label, embeddings in embeddings_dict.items():
            prototype, threshold = self.calculate_prototype(embeddings)

            results[label] = {
                
            }