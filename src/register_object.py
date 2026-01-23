import numpy as np
import cv2
import os

from datetime import datetime
from typing import Tuple

from src.utils.system_mode import SystemMode

class RegisterObject:
    def __init__(self, config, detector, extractor, memory):
        self.config = config
        self.detector = detector
        self.extractor = extractor
        self.memory = memory
        
        self.text_finish_instruction = False
        
    def register(self, frame: np.ndarray, key: int, object_name: str) -> Tuple[np.ndarray, SystemMode]:
        """
        Interactive function to capture views of an object (90º rotations).

        Args:
            frame (np.ndarray): camera frame coming from CameraStream.
            object_name: Name of the object that will be registered.
        """        
        # Detect object
        box = self.detector.detect(frame)

        # Display
        display_frame = frame.copy()
        
        # Show texts
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, "Press [Space] to capture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No object detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if self.text_finish_instruction:
            cv2.putText(display_frame, "When ready, press [F] to finish", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Keyboard input        
        if key == 32 and box is not None: # 32 = [Space]
            # Capture and crop
            crop = self.detector.crop_object(frame, box)
            
            # Create folder to save
            raw_images_dir = self.config['paths']['raw_images']
            object_dir = os.path.join(raw_images_dir, object_name)
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
            
            self.text_finish_instruction = True
            
        elif key == ord('f'):
            print("\n"
                  "Processing captures...")
            
            captures = self.memory.get_raw_images_of_object(object_name)

            # Extract embeddings from all 4 captures
            embeddings = self.extractor.extract_batch(captures)

            # Add to memory
            self.memory.add_object(object_name, embeddings)

            # Verify it was saved
            prototypes, labels = self.memory.get_all_prototypes()
            if object_name in labels:
                print(f"Sucessfully registered '{object_name}'.")
                print(f"  Database now has {len(labels)} {'object' if len(labels) == 1 else 'objects'}.")
            else:
                print("Failed to save object to database.")
        
            print("Registration finished.")
            
            self.text_finish_instruction = False
            
            return display_frame, SystemMode.RECOGNIZE
            
        return display_frame, SystemMode.REGISTER
