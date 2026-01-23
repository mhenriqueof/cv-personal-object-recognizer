import cv2
import numpy as np

class RecognizeObject:
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        
        # Get recognition thresholds
        self.threshold_high = self.config['recognition']['similarity_threshold_high']
        self.threshold_low = self.config['recognition']['similarity_threshold_low']

        # Smooth display
        self.current_label = "Initializing..."
        self.current_similarity = 0.0
        self.last_recognition_time = 0

    def recognize(self, frame: np.ndarray) -> np.ndarray:
        """
        Real-time object recognition from webcam.
        """
        # 1. Detect object
        box = self.detector.detect(frame)

        display_frame = frame.copy()

        if box is not None:
            x1, y1, x2, y2 = box
            
            # 2. Crop object
            crop = self.detector.crop_object(frame, box)

            # 3. Extract embedding
            embedding = self.extractor.extract(crop)

            # 4. Get all prototypes from database
            prototypes, labels = self.memory.get_all_prototypes()

            if len(prototypes) > 0:
                # 5. Calculate similarities
                ## Convert prototype to numpy array
                prototypes_array = np.array(prototypes)

                # Cosine similarity: dot product for normalized vectors
                similarities = np.dot(prototypes_array, embedding)

                # 6. Find best match
                best_idx = np.argmax(similarities)
                best_similarity = similarities[best_idx]
                best_label = labels[best_idx]

                # 7. Decision logic
                if best_similarity >= self.threshold_high:
                    current_label = best_label
                    current_similarity = best_similarity
                    color = (0, 255, 0) # Green - recognized
                    text = f"I'm seeing {current_label}"
                elif best_similarity >= self.threshold_low:
                    current_label = best_label
                    current_similarity = best_similarity
                    color = (0, 255, 255) # Yellow - uncertain
                    text = f"I think I'm seeing {current_label}"
                else:
                    current_similarity = best_similarity
                    color = (0, 0, 255)
                    text = f"I don't know this object"
            else:
                # No objects in database
                text = "No objects in database"
                current_similarity = 0.0
                color = (255, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Show label and similarity
            label_text = f"{text}, I'm {current_similarity * 100 :.2f}% sure."
            cv2.putText(display_frame, label_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        else:
            # No object detected
            cv2.putText(display_frame, "No object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return display_frame
