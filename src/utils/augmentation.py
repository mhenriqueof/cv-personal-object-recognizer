import numpy as np

from src.utils.setups import setup_logger
from src.utils.loads import load_config
from typing import List

class AugmentationPipeline:
    """Generates augmented version of an object crop to increase robustness."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.aug_strength = self.config['registration']['augmentation_strength']
        self.num_aug = self.config['registration']['num_augmentations_per_image']
        self.logger.info(f"Augmentation pipeline initialized (strength: {self.aug_strength})")
        
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generate augmented versions of a single image.
        
        Args:
            image: BGR image array (H, W, C).
        
        Returns:
            List of augmented BGR images (including the original).
        """
        