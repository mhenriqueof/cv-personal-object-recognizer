import cv2
import numpy as np

class RecognizeObject:
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        
        # Colors for different objects
        self.colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (255, 0, 255),  # Magenta
            (0, 165, 255),  # Orange
        ]
        
        # Get recognition settings
        self.max_objects = self.config['recognition']['max_objects']
        self.threshold_high = self.config['recognition']['similarity_threshold_high']
        self.threshold_low = self.config['recognition']['similarity_threshold_low']

    def recognize(self, frame: np.ndarray) -> np.ndarray:
        """
        Recognizes multiple objects in frame.
        
        Returns:
            Frame with visualizations drawn.
        """
        # Frame to work on
        display_frame = frame.copy()
        
        # Show instructions
        instructions_text = "[R] to register new object, [C] to clean database"
        cv2.putText(display_frame, instructions_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)    
        
        # Detect objects
        boxes = self.detector.detect(frame, max_objects=self.max_objects)
        if not boxes:
            return display_frame
        
        # Get database prototypes
        prototypes, labels = self.memory.get_all_prototypes()

        if len(prototypes) == 0:
            cv2.putText(display_frame, "No objects in database", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Still show detections in gray
            for i, box in enumerate(boxes):
                color = (128, 128, 128) # Gray for unknown
                self._draw_detection(display_frame, box, f"Object {i+1}", 0.0, color)
            return display_frame
        
        # Convert prototypes to numpy array for batch comparasion
        prototypes_array = np.array(prototypes)

        # Process each detected object
        for i, box in enumerate(boxes):
            # Crop object
            crop = self.detector.crop_object(frame, box)

            # Extract embedding
            embedding = self.extractor.extract(crop)

            # Compare with all prototypes
            similarities = np.dot(prototypes_array, embedding)

            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            best_label = labels[best_idx]

            # Choose color for this object
            color = self.colors[i % len(self.colors)]

            # Decision logic
            if best_similarity >= self.threshold_high:
                label_text = f"{best_label} ({best_similarity:.2f})"
                thickness = 2
            elif best_similarity >= self.threshold_low:
                color = (0, 255, 255)  # Yellow
                label_text = f"{best_label} ({best_similarity:.2f})"
                thickness = 1
            else:
                color = (0, 0, 255) # Red
                label_text = f"Unknown ({best_similarity:.2f})"
                thickness = 1
                
            # Draw bounding box and label
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Put label above box
            label_y = max(20, y1 - 10)
            cv2.putText(display_frame, label_text, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)        

        return display_frame

    def _draw_detection(self, frame, box, label, similarity, color):
        """Helps to draw a single detection."""
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label} ({similarity:.2f})"
        label_y = max(20, y1 - 10)
        cv2.putText(frame, label_text, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
