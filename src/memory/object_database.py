import os
import numpy as np
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.utils.setups import setup_logger
from src.utils.loads import load_config

class ObjectDatabase:
    """Manages storage and retrieval of object prototypes in a JSON database."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.db_path = self.config['paths']['objects_database']
        self.raw_images_dir = self.config['paths']['raw_images_dir']

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.raw_images_dir, exist_ok=True)

        # Load existing database or create empty
        self.database = self._load_database()
        self.logger.info(f"Object database initialized at '{self.db_path}'.")
        self.logger.info(f"Currently storing {len(self.database)} objects.")

    def _load_database(self) -> Dict[str, Any]:
        """Loads database from JSON file."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                    # Convert prototypes back to numpy arrays
                    for obj_id, obj_data in data.items():
                        if 'prototype' in obj_data:
                            obj_data['prototype'] = np.array(obj_data['prototype'])
                    return data
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load database: {e}. Starting fresh.")
                return {}
        return {}
        
    def _save_database(self) -> None:
        """Saves database to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        save_data = {}
        for obj_id, obj_data in self.database.items():
            save_data[obj_id] = obj_data.copy()
            if 'prototype' in save_data[obj_id]:
                save_data[obj_id]['prototype'] = save_data[obj_id]['prototype'].tolist()
            
        try:
            with open(self.db_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            self.logger.debug(f"Database saved with {len(self.database)} objects.")
        except IOError as e:
            self.logger.error(f"Failed to save database: {e}")
            
    def add_object(self, label: str, prototype: np.ndarray, threshold: float,
                   metadata: Optional[Dict] = None) -> str:
        """
        Adds a new object to the database.
        
        Args:
            label: Human-readable label for the object.
            prototype: Normalized prototype vector.
            threshold: Distance threshold for recognition.
            metadata: Optional additional information.
            
        Returns:
            Object ID (unique identifier).
        """
        # Generate a unique ID (timestamp + label hash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        obj_id = f"{timestamp}_{hash(label) % 10000:04d}"

        self.database[obj_id] = {
            'label': label,
            'prototype': prototype.copy(),
            'threshold': float(threshold),
            'created_at': timestamp,
            'updated_at': timestamp,
            'metadata': metadata or {}
        }
        
        self._save_database()
        self.logger.info(f"Added object '{label}' with ID {obj_id}.")
        return obj_id
    
    def update_object(self, obj_id: str, prototype: np.ndarray, threshold: float, label: str) -> None:
        """Updates an existing object's data."""
        if obj_id not in self.database:
            raise KeyError(f"Object ID '{obj_id}' not found in database.")
        
        if prototype is not None:
            self.database[obj_id]['prototype'] = prototype.copy()
        if threshold is not None:
            self.database[obj_id]['threshold'] = float(threshold)
        if label is not None:
            self.database[obj_id]['label'] = label
        
        self.database[obj_id]['updated_at'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_database()
        self.logger.info(f"Updated object {obj_id}.")

    def delete_object(self, obj_id: str):
        """Removes an object from the database."""
        if obj_id in self.database:
            label = self.database[obj_id]['label']
            del self.database[obj_id]
            self._save_database()
            self.logger.info(f"Deleted object {obj_id} ('{label}').")
        else:
            self.logger.warning(f"Object ID '{obj_id}' not found for deletion.")

    def get_all_objects(self) -> Dict[str, Dict[str, Any]]:
        """Returns the entire database."""
        return self.database.copy()

    def get_object(self, obj_id: str) -> Optional[Dict[str, Any]]:
        """Gets specific object data."""
        return self.database.get(obj_id, None)
    
    def find_by_label(self, label: str) -> List[str]:
        """Finds object IDs by label (exact match)."""
        return [obj_id for obj_id, data in self.database.items() if data['label'].lower() == label.lower()]
    
    def clear(self) -> None:
        """Clear the entire database."""
        count = len(self.database)
        self.database = {}
        self._save_database()
        self.logger.warning(f"Cleared entire database ({count} objects removed).")
    
    def get_prototypes_matrix(self) -> Tuple[np.ndarray, List[str], List[float]]:
        """
        Gets all prototypes as a matrix for efficient similarity computation.
        
        Returns:
            prototype_matrix: (n_objects, embedding_dim) array.
            labels: List fo object labels in same order.
            thresholds: List of distance thresholds in same order.
        """
        prototypes = []
        labels = []
        thresholds = []

        for obj_id, obj_data in self.database.items():
            prototypes.append(obj_data['prototype'])
            labels.append(obj_data['label'])
            thresholds.append(obj_data['threshold'])

        if not prototypes:
            return np.array([]), [], []

        return np.stack(prototypes), labels, thresholds
    
# Quick test
if __name__ == '__main__':
    db = ObjectDatabase()

    # Create dummy prototype
    dummy_prototype = np.random.randn(576)
    dummy_prototype = dummy_prototype / np.linalg.norm(dummy_prototype)

    # Add a test object
    obj_id = db.add_object(
        label="Test Mug",
        prototype=dummy_prototype,
        threshold=0.3,
        metadata={"color": "blue", "owner": "Henrique"}
    )
    
    print(f"Added object with ID: {obj_id}.")
    print(f"Database now has {len(db.get_all_objects())} objects.")
    
    # Retrieve it
    obj_data = db.get_object(obj_id)
    if obj_data:
        print(f"Retrieved object label: {obj_data['label']}.")
        print(f"Prototype shape: {obj_data['prototype'].shape}")
    
    # Test prototypes matrix
    prototypes, labels, thresholds = db.get_prototypes_matrix()
    print(f"Prototypes matrix shape: {prototypes.shape if len(prototypes) > 0 else 'empty'}")

    # Clean up test object
    db.delete_object(obj_id)
    print(f"After cleanup: {len(db.get_all_objects())} objects.")
    print("Object database test: OK")
    