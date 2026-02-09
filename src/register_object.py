import numpy as np
import cv2
import os

from datetime import datetime
from typing import Tuple

from src.utils.input_object_name import input_object_name
from src.utils.system_mode import SystemMode
from src.utils.augmentation import Augmentation

class RegisterObject:
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        self.augmenter = Augmentation()
        
        self.object_name = None
        
        self.finish_instruction = False
        
    def register(self, frame: np.ndarray, key: int) -> Tuple[np.ndarray, SystemMode]:
        """
        Interactive function to capture views of an object (90º rotations).

        Args:
            frame (np.ndarray): camera frame coming from CameraStream.
            object_name: Name of the object that will be registered.
        """
        if self.object_name is None:
            self.object_name = input_object_name()
        
        # Detect object
        boxes = self.detector.detect(frame, max_objects=1)
        if boxes:
            box = boxes[0]
        else:
            box = None

        # Frame to work on
        display_frame = frame.copy()
        
        # Show texts in window
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press [Space] to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if self.finish_instruction:
            instruction_text = "Press [F] to finish (only one capture is enough)"
            cv2.putText(display_frame, instruction_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Keyboard input        
        if key == 32 and box is not None: # 32 = [Space]
            # Capture and crop
            crop = self.detector.crop_object(frame, box)
            
            # Create folder to save
            raw_images_dir = self.config['paths']['raw_images']
            object_dir = os.path.join(raw_images_dir, self.object_name)
            os.makedirs(object_dir, exist_ok=True)
            
            # Save captured image
            time_now = datetime.now()
            image_path = os.path.join(object_dir, f"{time_now.strftime('%Y_%m_%d_%H_%M_%S')}.jpg")
            success = cv2.imwrite(image_path, crop)
            
            # Check if image was saved
            if success:
                print(f" Saved: {image_path}")
            else:
                print(" Error: could not save the image.")

            # Show captured image briefly
            cv2.imshow("Captured", crop)
            cv2.waitKey(1000)
            cv2.destroyWindow("Captured")
            
            self.finish_instruction = True
            
        elif key == ord('f'):
            print("\n"
                  "Processing captures...")
            
            # Get all raw images
            raw_captures = self.memory.get_raw_images_of_object(self.object_name)
            
            # Augment raw images
            all_images = self.augmenter.augment_batch(raw_captures)

            # Extract embeddings from all
            embeddings = self.extractor.extract_batch(all_images)

            # Add to memory
            self.memory.add_object(self.object_name, embeddings)

            # Verify it was saved
            prototypes, labels = self.memory.get_all_prototypes()
            if self.object_name in labels:
                print(f"Sucessfully registered '{self.object_name}'.")
                print(f"  Database now has {len(labels)} {'object' if len(labels) == 1 else 'objects'}.")
            else:
                print("Failed to save object to database.")
        
            print("Registration finished.")
            
            # Save augmented images
            self.memory.save_augmented_images(self.object_name, all_images)

            self.object_name = None
            self.finish_instruction = False
            
            return display_frame, SystemMode.RECOGNIZE
            
        return display_frame, SystemMode.REGISTER
