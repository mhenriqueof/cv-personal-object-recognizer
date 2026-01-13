import cv2
import os

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

def capture_4_views(object_name: str, camera_idx: int = 0, save_images: bool = False) -> bool:
    """
    Interactive function to capture 4 views of an object (90º rotations).

    Args:
        object_name: Name of the object that will be registered.
        camera_idx: The index of the camera used (default = 0).
    """
    
    config = load_config()
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    memory = MemoryManager()

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return False
    
    print("\n"
          f"Registration for: '{object_name}'")
    print("-" * 50)
    print("Instructions:")
    print("1. Hold the object in front of the camera.")
    print("2. Press [Space] to capture when the object is detected.")
    print("3. Rotate object ~90º after each capture.")
    print("4. Press [Q] to quit early.")
    print("-" * 50)

    captures = []
    angles = ["0º (front)", "90º", "180º (back)", "270º"]
    
    for view_idx, angle in enumerate(angles, start=1):
        print("\n"
              f"View {view_idx}/4: {angle}")
        
        captured = False
        while not captured:
            ok, frame = cap.read()
            if not ok:
                print(f"Failed to read from camera with index {camera_idx}.")
                break
            
            # Detect object
            box = detector.detect(frame)

            # Display
            display_frame = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"View {view_idx}: {angle}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, "Press [Space] to capture",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "No object detected - move closer",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
            cv2.imshow(f"Register: {object_name}", display_frame)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Registration cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            elif key == 32 and box is not None: # 32 = [Space]
                # Capture and crop
                crop = detector.crop_object(frame, box)
                captures.append(crop)
                captured = True
                
                # Save captured image
                if save_images:
                    raw_images_dir = config['paths']['raw_images']
                    object_dir = os.path.join(raw_images_dir, object_name)
                    os.makedirs(os.path.dirname(object_dir), exist_ok=True)

                    image_path = os.path.join(object_dir, f'{view_idx:02d}.jpg')
                    cv2.imwrite(image_path, crop)
                    print(f" Saved: {image_path}")

                # Show captured image briefly
                cv2.imshow("Captured", crop)
                cv2.waitKey(500)
                cv2.destroyWindow("Captured")

                print(f" Captured view {view_idx}")

    cap.release()
    cv2.destroyAllWindows()

    if len(captures) != 4:
        print(f"Only captures {len(captures)} images (need 4).")
        return False
    
    print("\n"
          "Processing captures...")

    # Extract embeddings from all 4 captures
    embeddings = extractor.extract_batch(captures)

    # Add to memory
    memory.add_object(object_name, embeddings)

    # Verify it was saved
    prototypes, labels = memory.get_all_prototypes()
    if object_name in labels:
        print(f"Sucessfully registered '{object_name}'.")
        print(f"  Database now has {len(labels)} {'object' if len(labels) == 1 else 'objects'}.")
        return True
    else:
        print("Failed to save object to database.")
        return False
    
def main():
    print("\n" +
          "-" * 50)
    print("Personal Object Registration")
    print("-" * 50)

    # Get object name
    object_name = input("\n"
                        "Enter the object name (e.g. \"My Mug\"): ").strip()
    if not object_name:
        print("Object name cannot be empty.")
        return
    
    # Run capture
    success = capture_4_views(object_name, save_images=True)

    if success:
        print("\n"
              "Registration complete!")
        print("You can now run recognition mode.")
    else:
        print("Registration failed.")

if __name__ == '__main__':
    main()
