import cv2
import numpy as np

from typing import Tuple

from src.utils.system_mode import SystemMode

class RecognizeObject:
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
        self.detection_interval = 3 # Detect every 3rd frame
        self.frame_counter = 0
        self.last_boxes = [] # Check last detection

    def recognize(self, frame: np.ndarray, key: int) -> Tuple[np.ndarray, SystemMode]:
        """
        Recognizes multiple objects in frame.
        
        Returns:
            Frame with visualizations drawn.
        """
        display_frame = frame.copy()
        
        # Smart Detection: Only run detection every N frames
        self.frame_counter += 1
        
        if self.frame_counter % self.detection_interval == 0 or not self.last_boxes:
            # Run fresh detection
            boxes = self.detector.detect(frame, max_objects=self.max_objects)
            self.last_boxes = boxes
        else:
            # Use cached detection
            boxes = self.last_boxes
                    
        # Show instructions
        instructions_text = "[R] register new object [C] clean database [Q] Quit"
        cv2.putText(display_frame, instructions_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if key == ord('r'):
            return display_frame, SystemMode.REGISTER
        elif key == ord('c'):
            self.memory.clear()
        
        # Get database prototypes
        prototypes, labels = self.memory.get_all_prototypes()

        if len(prototypes) == 0:
            cv2.putText(display_frame, "No objects in database", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Still show detections in gray
            for i, box in enumerate(boxes):
                color = (11, 11, 11) # Black for unknown
                self._draw_detection(display_frame, box, f"Object {i+1}", color)
            return display_frame, SystemMode.RECOGNIZE
    
        # Batch crop all objects
        crops = []
        valid_boxes = []

        for box in boxes:
            crop = self.detector.crop_object(frame, box)
            # Skip very small crops
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                crops.append(crop)   
                valid_boxes.append(box)
                
        if not crops:
            return display_frame, SystemMode.RECOGNIZE
        
        # Batch extraction (single model pass for all crops)
        embeddings = self.extractor.extract_batch(crops)
        
        # Convert prototypes to numpy array for batch comparasion
        prototypes_array = np.array(prototypes)
        
        # Batch Similarity Computation (Matrix multiplication)
        # embeddings shape: (n_objects, 576)
        # prototypes_array shape: (n_prototypes. 576)
        # similarities shape: (n_objects, n_prototypes)
        similarities = np.dot(embeddings, prototypes_array.T)

        # Process results
        for i, (box, object_similarities) in enumerate(zip(valid_boxes, similarities)):
            # Find best match
            best_idx = np.argmax(object_similarities)
            best_similarity = object_similarities[best_idx]
            best_label = labels[best_idx]

            # Choose color for this object

            # Decision logic
            if best_similarity >= self.threshold_high:
                color = (0, 255, 0)  # Green
                label_text = f"{best_label} ({best_similarity:.2f})"
            elif best_similarity >= self.threshold_low:
                color = (0, 255, 255)  # Yellow
                label_text = f"{best_label} ({best_similarity:.2f})"
            else:
                color = (11, 11, 11) # Black for unknown
                label_text = f"Object {i+1}"
                
            # Draw bounding box and label
            self._draw_detection(display_frame, box, label_text, color)

        return display_frame, SystemMode.RECOGNIZE

    def _draw_detection(self, frame, box, label_text, color):
        """Helps to draw a single detection."""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_y = max(20, y1 - 10)
        cv2.putText(frame, label_text, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
