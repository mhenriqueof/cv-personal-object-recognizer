import numpy as np
import torch
from typing import Tuple, List
from ultralytics import YOLO # type: ignore

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

class ObjectDetector:
    """
    YOLO-based object detector for finding prominent objects in frames.
    
    Uses YOLO for detection, filters out unwanted classes (persons) and returns sorted
    bounding boxes.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()
        
        # Set all random seeds for reproducibility
        set_seed()
        
        # Check GPU available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading detector on {self.device}.")
        
        # Load YOLO Nano model
        self.model_name = self.config['detector']['model_name']
        self.model = YOLO(self.model_name)
        self.confidence_threshold = self.config['detector']['confidence_threshold']
        
        self.logger.info(f"YOLO detector initialized '{self.model_name}'.")

    def detect(self, frame: np.ndarray, max_objects: int = 3) -> List[Tuple[int, int, int, int]]:
        """
        Detects multiple objects using YOLO.
        
        Args:
            frame: BGR image from OpenCV.
            max_objects: Maximum number of objects to return.
            
        Returns:
            List of bounding box (x1, y1, x2, y2) sorted by size (largest first).
            Empty list if no objects detected.
        """
        # Run YOLO inference
        results = self.model(frame,
                             conf=self.confidence_threshold,
                             iou=0.45,
                             verbose=False,
                             device=self.device)

        if len(results) == 0 or results[0].boxes is None:
            return []
        
        # Get boxes, scores, labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        
        if len(boxes) == 0:
            return []
        
        # Filter out unwanted classes (person = 0)
        unwanted_classes = [0]
        valid_mask = ~np.isin(labels, unwanted_classes)

        if not np.any(valid_mask):
            return []
                    
        # Get boxes excluding person
        boxes_filtered = boxes[valid_mask]
        
        # Calculate areas for sorting
        areas = (boxes_filtered[:, 2] - boxes_filtered[:, 0]) * \
                (boxes_filtered[:, 3] - boxes_filtered[:, 1])

        # Sort by area (largest first)
        sorted_indices = np.argsort(areas)[::-1]

        # Take top max_objects
        top_indices = sorted_indices[:max_objects]

        valid_boxes = []
        h, w = frame.shape[:2]

        for idx in top_indices:
            box = boxes_filtered[idx].astype(int)
            x1, y1, x2, y2 = box

            # Clamp to frame boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(1, min(x2, w))
            y2 = max(1, min(y2, h))

            # Skip if too small
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue
            
            valid_boxes.append((x1, y1, x2, y2))

            if len(valid_boxes) >= max_objects:
                break
            
        self.logger.debug(f"Detected {len(valid_boxes)} objects.")
        return valid_boxes

    def crop_object(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Crops the frame using the bounding box."""
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]
    