import cv2
import numpy as np

from typing import List

from src.utils.config import load_config
from src.utils.logger import setup_logger

class Augmentation:
    """
    Simple brightness augmentantation for object registration.
    
    Generates brighter and darker version of input images to improve robustness 
    to lighting conditions.
    """
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger(self.__class__.__name__)
        
        # Default parameter
        self.aug_strength = self.config['registration']['augmentation_strength']
        
        self.logger.info(f"Augmentation initialized (strength: {int(self.aug_strength * 100)}%).")
        
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generates augmentations for a single image.
        
        Strategy: Creates 2 new images:
        1. Original + brightness up
        2. Original + brightness down
        
        Args:
            image: BGR image.
        
        Returns:
            List of 2 augmented BGR images.
        """
        bright_up = self._adjust_brightness(image, +self.aug_strength)
        bright_down = self._adjust_brightness(image, -self.aug_strength)
            
        self.logger.debug(f"Generated 2 augmented images.")
        
        return [bright_up, bright_down]
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness by multiplicative factor.
        
        Args:
            image: BGR image.
            factor: Positive = brighter, Negative = darker
            
        Returns:
            Brightness-adjusted image
        """
        # Convert to float for calculation
        img_float = image.astype(np.float32)

        # Adjust brightness (additive)
        adjusted = img_float * (1.0 + factor)

        # Clip to valid range and convert back
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Augment a batch of images.

        Args:
            images: List of BGR images.
            
        Returns:
            Flattened list of all augmented images.
        """
        all_augmented = []

        for img in images:
            all_augmented.extend(self.augment(img))
            
        self.logger.info(f"Batch augmentation: {len(images)} images → "
                         f"{len(all_augmented)} total images.")

        return all_augmented
    