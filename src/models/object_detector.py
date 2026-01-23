import numpy as np
import torch
import cv2
from typing import Optional, Tuple
from ultralytics import YOLO # type: ignore

from src.utils.config import load_config
from src.utils.logger import setup_logger

class ObjectDetector:
    """Responsible for object detector to find the most prominent object in a frame."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading detector on {self.device}.")
        
        # Load YOLO Nano model
        self.model_name = self.config['detector']['model_name']
        self.model = YOLO(self.model_name)
        
        self.confidence_threshold = self.config['detector']['confidence_threshold']
        self.logger.info(f"YOLO detector initialized '{self.model_name})'.")

    def detect(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detects objects and returns the bounding box of the most confident detection.
        
        Args:
            frame: BGR image from OpenCV.
            
        Returns:
            Optional bounding box (x1, y1, x2, y2) in original coordinates, or None if no
            confident detection.
        """
        # Run YOLO inference
        results = self.model(frame,
                             conf=self.confidence_threshold,
                             verbose=False,
                             device=self.device)

        if len(results) == 0 or results[0].boxes is None:
            return None
        
        
        # Get boxes, scores, labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        scores = results[0].boxes.conf.cpu().numpy()
        labels = results[0].boxes.cls.cpu().numpy()
        
        if len(boxes) == 0:
            return None
        
        # 1. Filter by confidence
        is_confident = (scores >= self.confidence_threshold)
        # 2. Filter by label: exclude the 'person' label (COCO label index 1)
        is_not_person = (labels != 1)

        # Combine the masks to get valid indices
        valid_indices = is_confident & is_not_person
        
        if not np.any(valid_indices):
            return None
        
        # Filter out 'person' class
        person_indices = np.where(labels == 0)[0]
        valid_indices = np.setdiff1d(np.arange(len(boxes)), person_indices)
        
        if len(valid_indices) == 0:
            return None
        
        # Get boxes excluding persons
        boxes_no_person = boxes[valid_indices]
        scores_no_person = scores[valid_indices]
        
        best_idx = np.argmax(scores_no_person)

        best_box = boxes_no_person[best_idx].astype(int)
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
    