import numpy as np
import cv2
import os

from datetime import datetime
from typing import Tuple

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

config = load_config()
detector = ObjectDetector()
extractor = FeatureExtractor()
memory = MemoryManager()

def register_object(frame: np.ndarray, key: int, object_name: str) -> Tuple[np.ndarray, bool]:
    """
    Interactive function to capture views of an object (90º rotations).

    Args:
        frame (np.ndarray): camera frame coming from CameraStream.
        object_name: Name of the object that will be registered.
    """        
    # Detect object
    box = detector.detect(frame)

    # Display
    display_frame = frame.copy()
    
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, "Press [Space] to capture", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No object detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Keyboard input        
    if key == 32 and box is not None: # 32 = [Space]
        # Capture and crop
        crop = detector.crop_object(frame, box)
        
        # Create folder to save
        raw_images_dir = config['paths']['raw_images']
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
        
    elif key == ord('f'):
        print("\n"
              "Processing captures...")
        
        captures = memory.get_raw_images_of_object(object_name)

        # Extract embeddings from all 4 captures
        embeddings = extractor.extract_batch(captures)

        # Add to memory
        memory.add_object(object_name, embeddings)

        # Verify it was saved
        prototypes, labels = memory.get_all_prototypes()
        if object_name in labels:
            print(f"Sucessfully registered '{object_name}'.")
            print(f"  Database now has {len(labels)} {'object' if len(labels) == 1 else 'objects'}.")
        else:
            print("Failed to save object to database.")
    
        print("Registration finished.")
        
        return display_frame, True
        
    return display_frame, False
