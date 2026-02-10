import shutil
import cv2
import os
import json
import numpy as np

from typing import Dict, List, Tuple
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.config import load_config

class MemoryManager:
    """
    Manages object prototypes and raw images storage.
    
    Stores objects prototypes (average embeddings) in JSON database.
    Manages raw and augmented images in filesystem.
    """
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
            embeddings: numpy array of shape (n_samples, embedding_dim).
        """
        if embeddings.size == 0:
            self.logger.error("No embeddings provided.")
            return
        
        # Calculate prototype (average of embeddings)
        prototype = np.mean(embeddings, axis=0)
        # Re-normalize the prototype
        norm = np.linalg.norm(prototype)
        if norm > 0:
            prototype = prototype / norm
        else:
            self.logger.error("Zero-norm prototype, cannot add object.")
            return
        
        self.database[label] = {
            'prototype': prototype,
            'num_images': embeddings.shape[0]
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
    
    def get_raw_images_of_object(self, object_name: str) -> List:
        """
        Get all raw images of an object.
        
        Args:
            object_name (str): name of the object, will be searched its corresponding folder.
        
        Returns:
            captures: list of images in np.ndarray.        
        """
        images_list = []

        # Object folder path
        raw_images_dir = self.config['paths']['raw_images']
        object_path = Path(raw_images_dir) / object_name
        if not object_path.exists():
            print(f" Error: Could not find '{object_path}' folder.")
            return images_list

        # Get images
        for file_path in object_path.glob("*.jpg"):
            img = cv2.imread(str(file_path))
            
            if img is not None:
                images_list.append(img)
            else:
                print(f" Could not read the image '{file_path}'.")
            
        print(f"All images of object '{object_name}' loaded.")
        
        return images_list
        
    def save_augmented_images(self, object_name: str, images: List[np.ndarray]) -> None:
        """Save augmented images."""
        save_dir = self.config['paths']['raw_images']
        object_path = Path(save_dir) / object_name
        
        if not object_path.exists():
            print(f" Error: Could not find '{object_path}' folder.")
            return
        
        for i, img in enumerate(images):
            if i != 0:
                cv2.imwrite(f"{object_path}/augment_{i:02d}.jpg", img)

        print(f" Saved {len(images)} images to {save_dir}/")
        
    def delete_object(self, object_name: str) -> None:
        """
        Completely delete an object (prototype + images).
        
        Args:
            object_name: Object to delete.
            
        Returns:
            True if deleted successfully, False otherwise.
        """
        if object_name not in self.database:
            self.logger.warning(f"Object '{object_name}' not found in database.")
        
        # Delete from database
        del self.database[object_name]
        self._save_database()

        # Delete image folder
        raw_images_dir = self.config['paths']['raw_images']
        object_path = Path(raw_images_dir) / object_name
        
        if object_path.exists() and object_path.is_dir():
            try:
                shutil.rmtree(object_path)
                self.logger.info(f"Deleted old images for '{object_name}'.")
                self.logger.info(f"Deleted object '{object_name}' completely.")
            except Exception as e:
                self.logger.error(f"Failed to delete images for '{object_name}': {e}")
        else:
            self.logger.warning(f"Deleted '{object_name}' from database, but failed to delete images.")
                        
    def clear(self) -> None:
        """Clear all data."""
        # Clear JSON
        self.database = {}
        self._save_database()
        self.logger.info("Database cleared.")

        # Clear raw images
        raw_images_dir = self.config['paths']['raw_images']
        
        if os.path.exists(raw_images_dir):
            try:
                shutil.rmtree(raw_images_dir)
                print(" Raw images folder deleted.")
            except Exception as e:
                print(f" Error: Couldn't delete raw images folder: {e}")
        else:
            print(" Raw images folder doesn't exist.")
            