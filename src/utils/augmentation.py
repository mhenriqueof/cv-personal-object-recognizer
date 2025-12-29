import numpy as np

from src.utils.setups import setup_logger
from src.utils.loads import load_config
from typing import List
import random
import cv2

class AugmentationPipeline:
    """Generates augmented version of an object crop to increase robustness."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.aug_strength = self.config['registration']['augmentation_strength']
        self.num_aug = self.config['registration']['num_augmentations_per_image']
        self.logger.info(f"Augmentation pipeline initialized (strength: {self.aug_strength}).")
        
    def augment(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Generates augmented versions of a single image.
        
        Args:
            image: BGR image array (H, W, C).
        
        Returns:
            List of augmented BGR images (including the original).
        """
        augmented = [image.copy()] # start with original
        
        for i in range(self.num_aug):
            aug_img = image.copy()
            
            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            h, w = aug_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_img = cv2.warpAffine(aug_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

            # Random brightness/constrast
            alpha = 1.0 + random.uniform(-self.aug_strength, self.aug_strength) # constrast
            beta = random.uniform(-self.aug_strength * 50, self.aug_strength * 50) # brightness
            aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

            # Random slight zoom (90% to 110%)
            zoom = random.uniform(0.9, 1.1)
            new_h, new_w = int(h * zoom), int(w * zoom)
            aug_img = cv2.resize(aug_img, (new_w, new_h))

            # Crop back to original size (center crop)
            start_y = max(0, (new_h - h) // 2)
            start_x = max(0, (new_w - w) // 2)
            aug_img = aug_img[start_y:start_y+h, start_x:start_x+w]

            # Ensure size matches original (in case of rounding errors)
            if aug_img.shape[:2] != (h, w):
                aug_img = cv2.resize(aug_img, (w, h))
            
            # Random horizontal flip (50% chance)
            if random.random() > 0.5:
                aug_img = cv2.flip(aug_img, 1)

            augmented.append(aug_img)

        self.logger.debug(f"Generated {len(augmented)} images (1 original + {self.num_aug} augmented).")

        return augmented
    
    def augment_batch(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """Generates augmented versions for multiple base images."""
        all_augmented = []
        for img in images:
            all_augmented.extend(self.augment(img))
        return all_augmented
    
# Quick test
if __name__ == '__main__':
    # Load the test crop we saved earlier
    test_img = cv2.imread('test_crop.jpg')

    if test_img is None:
        # Create a dummy image if test_crop.jpg doesn't exist
        test_img = np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        cv2.imwrite('test_dummy.jpg', test_img)

    aug_pipeline = AugmentationPipeline()
    augmented_images = aug_pipeline.augment(test_img)

    print(f"Original image shape: {test_img.shape}")
    print(f"Number of augmented images: {len(augmented_images)}")
    
    # Save a sample augmentation to visually verify
    cv2.imwrite('test_augmented.jpg', augmented_images[1])
    print("Saved test_augmented.jpg for visual inspection.")
    