import numpy as np
import cv2
import os

from datetime import datetime
from typing import Tuple, Optional

from src.utils.input_helper import input_object_name
from src.utils.system_mode import SystemMode
from src.utils.augmentation import Augmentation

class RegisterObject:
    """
    Object registration module.
    
    Guides user through capturing images of an object, applies augmentations 
    and stores prototypes in database.
    """
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        self.augmenter = Augmentation()
        
        self.object_name = None
        self.finish_instruction = False
        self.captures_count = 0
        self.max_captures = config['registration']['max_captures']
        
    def process(self, frame: np.ndarray, key: int) -> Tuple[np.ndarray, SystemMode]:
        """
        Processes registration flow for a single frame.

        Args:
            frame: Camera frame.
            key: Keyboard input.
        
        Returns:
            (display_frame, next_mode)
        """
        # Initialize object name if not set
        if self.object_name is None:
            self.object_name = input_object_name()
            # Check if object already exists
            if self.object_name in self.memory.database:
                # Delete previous object data
                self.memory.delete_object(self.object_name)
                print(f"Object '{self.object_name}' already exists. Overwriting.")
        
        # Detect object
        boxes = self.detector.detect(frame, max_objects=1)
        box = boxes[0] if boxes else None

        # Create display frame
        display_frame = self._create_display_frame(frame.copy(), box)
        
        # Handle keyboard input
        return self._handle_input(display_frame, frame, key, box)

    def _create_display_frame(self, frame: np.ndarray, box: Optional[Tuple]) -> np.ndarray:
        """Adds registration UI elements to frame."""
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Press [Space] to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.finish_instruction:
            text = (f"Press [F] to finish (1 capture is enough, "
                    f"{self.captures_count}/{self.max_captures} taken)")
            cv2.putText(frame, text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _handle_input(self, display_frame: np.ndarray, original_frame: np.ndarray, key: int,
                      box: Optional[Tuple]) -> Tuple[np.ndarray, SystemMode]:
        """Handles keyboard input during registration."""
        # Space key - capture image
        if key == ord(' ') and box is not None and self.captures_count <= 4:
            self._capture_image(original_frame, box)
            self.captures_count += 1
            self.finish_instruction = True
            return display_frame, SystemMode.REGISTER
        
        # F key - finish registration
        elif key == ord('f') and self.captures_count > 0:
            return self._finish_registration(display_frame)
        
        # Q key - cancel registration
        elif key == ord('q'):
            print("Registration cancelled.")
            # Reset for next registration
            self._reset_variables()
            return display_frame, SystemMode.RECOGNIZE
            
        return display_frame, SystemMode.REGISTER
    
    def _capture_image(self, frame: np.ndarray, box: Tuple) -> None:
        """Captures and saves a single image."""
        crop = self.detector.crop_object(frame, box)

        # Create save directory
        raw_images_dir = self.config['paths']['raw_images']
        object_dir = os.path.join(raw_images_dir, self.object_name) # type: ignore
        os.makedirs(object_dir, exist_ok=True)

        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = os.path.join(object_dir, f"{timestamp}.jpg")

        if cv2.imwrite(image_path, crop):
            print(f" Saved: {os.path.basename(image_path)}")
        else:
            print(" Failed to save image.")
            
        # Show captured image briefly
        cv2.imshow("Captured", crop)
        cv2.waitKey(1500)
        cv2.destroyWindow("Captured")

    def _finish_registration(self, display_frame: np.ndarray) -> Tuple[np.ndarray, SystemMode]:
        """Completes the registration process."""
        # Get captured images
        raw_captures = self.memory.get_raw_images_of_object(self.object_name)

        if len(raw_captures) < 1:
            print(f" Need at least 1 capture, got {len(raw_captures)}")
            return display_frame, SystemMode.REGISTER
        
        # Augment images
        augmented_images = self.augmenter.augment_batch(raw_captures)
        print(f" Generated {len(augmented_images)} new images.")

        # 
        all_images = raw_captures + augmented_images
        
        # Extract embeddings
        embeddings = self.extractor.extract_batch(all_images)

        # Store in database
        self.memory.add_object(self.object_name, embeddings)
        
        # Save augmented images
        self.memory.save_augmented_images(self.object_name, augmented_images)
        
        # Verify
        prototypes, labels = self.memory.get_all_prototypes()
        if self.object_name in labels:
            print(f" Successfully registered '{self.object_name}'.")
            print(f" Database now has {len(labels)} object(s).")
        else:
            print(f" Failed to register '{self.object_name}'.")

        # Reset for next registration
        self._reset_variables()
        
        return display_frame, SystemMode.RECOGNIZE

    def _reset_variables(self) -> None:
        self.object_name = None
        self.finish_instruction = False
        self.captures_count = 0
        