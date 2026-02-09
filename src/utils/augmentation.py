import cv2
import numpy as np

from typing import List

from src.utils.config import load_config
from src.utils.logger import setup_logger

class Augmentation:
    """
    Deterministic image augmentation for object registration.
    Always generates the same augmented images for the same input.
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
        
        Strategy: Creates 6 new images (7 in total):
        1. Original
        2. Rotate (180º)
        4. Original + brightness up
        5. Original + brightness down
        6. Rotate + brightness up
        7. Rotate + brightness down
        
        Args:
            image: BGR image.
        
        Returns:
            List of 6 augmented BGR images (deterministic order).
        """
        original = image.copy()
        augmented_images = [original]

        # Rotate image (180º)
        rotated = cv2.flip(original, -1)
        augmented_images.append(rotated)

        # Generate brightness variations for all three base images
        base_images = [original, rotated]

        for base_img in base_images:           
            bright_up = self._adjust_brightness(base_img, +self.aug_strength)
            bright_down = self._adjust_brightness(base_img, -self.aug_strength)

            augmented_images.append(bright_up)
            augmented_images.append(bright_down)
            
        # Verify if have 9 images
        assert len(augmented_images) == 6, f"Expected 6 images, got {len(augmented_images)}"

        self.logger.debug(f"Generated {len(augmented_images)} augmented images "
                          "(1 original + 6 augmentations).")
        
        return augmented_images        
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness by factor (factor > 0: brighter, factor < 0: darker).
        Clips to valid range [0, 255].
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
            augmented = self.augment(img)
            all_augmented.extend(augmented)
            
        self.logger.info(f"Batch augmentation: {len(images)} images → "
                         f"{len(all_augmented)} total images.")

        return all_augmented
    