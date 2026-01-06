import os
import json
import numpy as np

from typing import Dict, List, Tuple

from src.utils.logger import setup_logger
from src.utils.config import load_config

class MemoryManager:
    """Storage for objects. Stores only the average embedding (prototype) for each object."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.db_path = self.config['paths']['objects_database']
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.database = self._load_database()
        self.logger.info(f"Memory initialized. Objects: {len(self.database)}")

    def _load_database(self) -> Dict[str, Dict]:
        """Loads database from JSON."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                # Convert lists back to numpy arrays
                for label, obj_data in data.items():
                    obj_data['prototype'] = np.array(obj_data['prototype'])
                return data
            except Exception as e:
                self.logger.warning(f"Could not load database: {e}")
        return {}

    def _save_database(self) -> None:
        """Saves database to JSON."""
        save_data = {}
        for label, obj_data in self.database.items():
            save_data[label] = {
                'prototype': obj_data['prototype'].tolist(),
                'num_images': obj_data.get('num_images', 0)
            }
            
        with open(self.db_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    def add_object(self, label: str, embeddings: np.ndarray) -> None:
        """
        Adds new object by calculating and storing its prototype (average).
        
        Args:
            label: Object name.
            embeddings: List of embedding vectors (should be 4, one every 90 degrees).
        """
        if not embeddings:
            self.logger.error("No embeddings provided.")
            return
        
        # Calculate prototype (average of embeddings)
        prototype = np.mean(embeddings, axis=0)
        # Re-normalize the prototype
        prototype = prototype / (np.linalg.norm(prototype) + 1e-9)

        if label in self.database:
            self.logger.warning(f"Object 'label' already exists. Overwriting.")
        
        self.database[label] = {
            'prototype': prototype,
            'num_images': len(embeddings)
        }
        
        self._save_database()
        self.logger.info(f"Added object '{label}' with prototype from {len(embeddings)} images.")

    def get_all_prototypes(self) -> Tuple[List[np.ndarray], List[str]]:
        """
        Get all prototypes and their labels.
        
        Returns:
            (prototypes, labels)
            prototypes: List of prototype vectors.
            labels: List of corresponding labels.
        """
        labels = []
        prototypes = []

        for label, obj_data in self.database.items():
            prototypes.append(obj_data['prototype'])
            labels.append(label)

        return prototypes, labels

    def clear(self) -> None:
        """Clear all data."""
        self.database = {}
        self._save_database()
        self.logger.info("Database cleared.")
