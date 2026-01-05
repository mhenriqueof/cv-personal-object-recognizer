import numpy as np
import cv2
import time

from typing import Optional, List
from src.utils.setups import setup_logger
from src.utils.loads import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.utils.augmentation import AugmentationPipeline
from src.memory.prototype_calculator import PrototypeCalculator
from src.memory.object_database import ObjectDatabase

class RegistrationManager:
    """Manages the complete registration pipeline for learning new objects."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        # Initialize all components
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.augmentor = AugmentationPipeline()
        self.calculator = PrototypeCalculator()
        self.database = ObjectDatabase()

        self.num_raw_captures = self.config['registration']['num_raw_captures']
        self.logger.info("Registration manager initialized.")

    def capture_raw_images(self, video_source: int = 0, 
                           capture_delay: float = 0.5) -> Optional[List[np.ndarray]]:
        """
        Capture raw images of an object from webcam.
        
        Args:
            video_source: Webcam index (usually 0).
            capture_delay: Seconds between captures.
            
        Returns:
            List of cropped object images, or None if failed.
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error("Could not open webcam.")
            return None
        
        self.logger.info(f"Press 'C' to capture. Need {self.num_raw_captures} captures. Press 'Q' to quit.")
        
        captures = []
        last_capture_time = 0
        
        while len(captures) < self.num_raw_captures:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Detect object
            box = self.detector.detect(frame) # type: ignore

            # Display
            display_frame = frame.copy() # type: ignore
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Captures: {len(captures)}/{self.num_raw_captures}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No object detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registration - Capture Raw Images", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.logger.warning("Registration cancelled by user.")
                break
            
            if key == ord('c') and box is not None:
                current_time = time.time()
                if current_time - last_capture_time > capture_delay:
                    # Crop and save
                    crop = self.detector.crop_object(frame, box)
                    captures.append(crop)
                    last_capture_time = current_time
                    self.logger.info(f"Capture {len(captures)}/{self.num_raw_captures} taken.")
            
        cap.release()
        cv2.destroyAllWindows()

        if len(captures) < self.num_raw_captures:
            self.logger.warning(f"Only captured {len(captures)} raw images (needed {self.num_raw_captures}).")
            return None
        
        self.logger.info(f"Successfully captured {len(captures)} raw images.")
        return captures
    
    def register_object(self, label: str, raw_images: List[np.ndarray]) -> bool:
        """
        Full registration pipeline for a new object.
        
        Args:
            label: Name for the object.
            raw_images: List of raw cropped images.
            
        Returns:
            True if registration successful.
        """
        if not raw_images:
            self.logger.error("No raw images provided.")
            return False
        
        self.logger.info(f"Starting registration for '{label}' with {len(raw_images)} raw images.")

        # Step 1: augment images
        self.logger.info("Augmenting images...")
        all_images = self.augmentor.augment_batch(raw_images)
        self.logger.debug(f"Generated {len(all_images)} total images (aftger augmentation).")
        
        # Step 2: extract embeddings
        self.logger.info("Extracting embeddings...")
        embeddings = self.extractor.extract_batch(all_images)
        self.logger.debug(f"Embeddings shape: {embeddings.shape}")

        # Step 3: calculate prototype and threshold
        self.logger.info("Calculating prototype...")
        prototype, threshold = self.calculator.calculate_prototype(embeddings)

        # Step 4: store in database
        self.logger.info("Storing in database...")
        self.database.add_object(
            label=label,
            prototype=prototype,
            threshold=threshold,
            metadata={
                'num_raw_images': len(raw_images),
                'num_total_images': len(all_images),
                'registration_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        
        self.logger.info(f"Registration complete for '{label}' (threshold: {threshold:.4f})")
        return True
    
    def interactive_registration(self):
        """Interactive registration via command line."""
        print("\n"
              "-" * 50)
        print("Object Registration Mode")
        print("-" * 50)
        
        label = input("Enter object label: ").strip()
        if not label:
            print("Label cannot be empty. Aborting.")
            return
        
        print("\n"
              f"Preparing to capture '{label}'...")
        print("Make sure the object is clearly visible in the webcam.")
        input("Press [Enter] when ready...")

        # Capture images
        raw_images = self.capture_raw_images()
        if raw_images is None:
            print("Failed to capture images. Aborting.")
            return
        
        # Register
        print("\n"
              "Processing...")
        success = self.register_object(label, raw_images)
        
        if success:
            print("\n"
                  f"Successfully registered '{label}'.")
            print("You can now run recognition mode.")
        else:
            print("\n"
                  f"Failed to register '{label}'.")

# Quick test
if __name__ == '__main__':
    manager = RegistrationManager()
    print("Registration manager test: OK")

    # Test without webcam using dummy images
    dummy_images = [np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8) for _ in range(3)]

    # Quick registration test
    print("Note: Full test requires webcam interaction.")
    print("Run `python -m src.memory.registration_manager` separately to test.")
    