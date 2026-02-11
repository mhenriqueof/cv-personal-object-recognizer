import cv2
import numpy as np

from typing import Tuple

from src.utils.system_mode import SystemMode

class RecognizeObject:
    """
    Object recognition module.
    
    Detects multiple objects in frame, extracts features, compares with stored prototypes
    and display results. Uses frame skipping and batch processing for performance.
    """   
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        
        # Get recognition settings
        self.max_objects = self.config['recognition']['max_objects']
        self.threshold_high = self.config['recognition']['similarity_threshold_high']
        self.threshold_low = self.config['recognition']['similarity_threshold_low']
        
        # Frame Skipping
        self.detection_interval = self.config['performance']['detection_interval']
        self.frame_counter = 0
        self.last_boxes = []

    def process(self, frame: np.ndarray, key: int) -> Tuple[np.ndarray, SystemMode]:
        """
        Processes recognition for a single frame.
        
        Args:
            frame: Camera frame.
            key: Keyboard input.
            
        Returns:
            (display_frame, next_mode)
        """
        display_frame = frame.copy()
        
        # Handle keyboard input
        if key == ord('r'):
            return display_frame, SystemMode.REGISTER
        elif key == ord('c'):
            self.memory.clear()
                
        # Get database info
        prototypes, labels = self.memory.get_all_prototypes()

        # Display instructions
        self._add_instructions_text(display_frame)
        # Update display with database info
        self._add_database_info(display_frame, len(labels))
        
        # Smart detection with frame skipping
        boxes = self._get_detections(frame)       
        if boxes:
            # Process detections
            display_frame = self._process_detections(display_frame, frame, boxes, prototypes, labels)

        return display_frame, SystemMode.RECOGNIZE
    
    def _get_detections(self, frame: np.ndarray):
        """Gets object detection with frame skipping optimization."""
        self.frame_counter += 1
        
        # Only run detection every N frames
        if (self.frame_counter % self.detection_interval == 0) or (not self.last_boxes):
            boxes = self.detector.detect(frame, max_objects=self.max_objects)
            self.last_boxes = boxes
            return boxes
        else:
            return self.last_boxes
        
    def _process_detections(self, display_frame: np.ndarray, original_frame: np.ndarray,
                            boxes, prototypes, labels) -> np.ndarray:
        """Processes all detections and draws results."""
        # Batch crop objects
        crops, valid_boxes = [], []
        for box in boxes:
            crop = self.detector.crop_object(original_frame, box)
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                crops.append(crop)
                valid_boxes.append(box)
        
        if not crops or not prototypes:
            # No valid crops or empty database
            for i, box in enumerate(boxes):
                self._draw_detection(display_frame, box,
                                     f"Object {i+1}", (50, 50, 50)) # Gray
            return display_frame
        
        # Batch extract embeddings
        embeddings = self.extractor.extract_batch(crops)
        prototypes_array = np.array(prototypes)

        # Batch similarity computation
        similarities = np.dot(embeddings, prototypes_array.T)

        # Process each detection
        for i, (box, object_sims) in enumerate(zip(valid_boxes, similarities)):
            best_idx = np.argmax(object_sims)
            best_similarity = object_sims[best_idx]
            best_label = labels[best_idx]

            # Determine recognition result
            label_text, color = self._get_recognition_result(best_label, best_similarity, i)

            self._draw_detection(display_frame, box, label_text, color)

        return display_frame
    
    def _get_recognition_result(self, label: str, similarity: float, object_idx: int) \
                                -> Tuple[str, Tuple]:
        """Determines label and color based on similarity."""
        if similarity >= self.threshold_high:
            return f"{label} ({similarity:.2f})", (0, 255, 0) # Green
        elif similarity >= self.threshold_low:
            return f"{label}? ({similarity:.2f})", (0, 255, 255) # Yellow
        else:
            return f"Object {object_idx+1}", (50, 50, 50) # Gray
        
    def _draw_detection(self, frame: np.ndarray, box: Tuple, label: str, color: Tuple) -> None:
        """Draws a single detection with bounding box and label."""
        x1, y1, x2, y2 = box

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label
        label_y = max(20, y1 - 10)
        cv2.putText(frame, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def _add_instructions_text(self, frame: np.ndarray) -> None:
        """Adds message to show instructions."""
        text = "[R] Register new object [C] Clean database [Q] Quit"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _add_database_info(self, frame: np.ndarray, num_objects: int) -> None:
        """Adds database information to display."""
        db_text = f"Objects in DB: {num_objects}"
        cv2.putText(frame, db_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
