import numpy as np
import torch
import torchvision
import cv2
from typing import Optional, Tuple
from torchvision.models.detection import ssdlite320_mobilenet_v3_large,SSDLite320_MobileNet_V3_Large_Weights

from src.utils import load_config, setup_logger

class ObjectDetector:
    """A lightweitght object detector to find the most prominent object in a frame."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading detector on {self.device}.")
        
        # Load pre-trained SSD with MobileNetV3 backbone
        self.model = ssdlite320_mobilenet_v3_large(
            weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

        # Preprocess transform
        self.transform = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.transforms()
        
        self.confidence_threshold = self.config['detector']['confidence_threshold']
        self.logger.info("Detector initialized.")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects objects and returns the bounding box of the most confident detection.
        
        Args:
            frame: BGR image from OpenCV.
            
        Returns:
            Optional bounding box (x1, y1, x2, y2) in original coordinates,
            or None if no confident detection.
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply transform and add batch dimension
        input_tensor = self.transform(torch.from_numpy(frame_rgb).permute(2, 0, 1))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(input_tensor)[0]

        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()

        # Filter by confidence and keep only "person" (1) and common objects (COCO labels)
        valid_indices = (scores >= self.confidence_threshold)
        
        if not np.any(valid_indices):
            return None
        
        # Get the index of the highest condifence detection
        best_idx = np.argmax(scores[valid_indices])
        best_box = boxes[valid_indices][best_idx].astype(int)

        # Convert to (x1, y1, x2, y2)
        x1, y1, x2, y2 = best_box
        
        # Ensure box is within frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Check if box is valid (non-zero area)
        if x2 <= x1 or y2 <= y1:
            return None
        
        self.logger.debug(f"Detected box: ({x1}, {y1}), ({x2}, {y2})")
        return (x1, y1, x2, y2)

    def crop_object(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Crops the frame using the bounding box."""
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]
    
# Test
if __name__ == '__main__':
    detector = ObjectDetector()
    print("Detector module test: OK")
