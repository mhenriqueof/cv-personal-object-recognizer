import cv2
import time
import numpy as np

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

def live_recognition():
    """
    Real-time object recognition from webcam.
    """
    config = load_config()
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    memory = MemoryManager()

    # Get recognition thresholds
    threshold_high = config['recognition']['similarity_threshold_high']
    threshold_low = config['recognition']['similarity_threshold_low']

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Error: could not open webcam.")
        return
    
    print("\n"
          " Starting Live Recognition")
    print("-" * 50)
    print("Press [Q] to quit")
    print("Press [R] to register new object")
    print("-" * 50)

    # FPS calculation
    fps_report_interval = 2.0
    last_fps_time = time.time()
    frame_count = 0
    
    # Smooth display
    current_label = "Initializing..."
    current_similarity = 0.0
    last_recognition_time = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print(" Failed to read from webcam.")
            break
        
        frame_count += 1
        
        # 1. Detect object
        box = detector.detect(frame)

        display_frame = frame.copy()

        if box is not None:
            x1, y1, x2, y2 = box
            
            # 2. Crop object
            crop = detector.crop_object(frame, box)

            # 3. Extract embedding
            embedding = extractor.extract(crop)

            # 4. Get all prototypes from database
            prototypes, labels = memory.get_all_prototypes()

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
                if best_similarity >= threshold_high:
                    current_label = best_label
                    current_similarity = best_similarity
                    color = (0, 255, 0) # Green - recognized
                    last_recognition_time = time.time()
                elif best_similarity >= threshold_low:
                    current_label = f"{best_label}?"
                    current_similarity = best_similarity
                    color = (0, 255, 255) # Yellow - uncertain
                else:
                    current_label = "Unknown"
                    current_similarity = best_similarity
                    color = (0, 0, 255)
            else:
                # No objects in database
                current_label = "No objects in database"
                current_similarity = 0.0
                color = (255, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label and similarity
            label_text = f"{current_label} ({current_similarity:.2f})"
            cv2.putText(display_frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw small preview of crop
            preview_size = 80
            crop_preview = cv2.resize(crop, (preview_size, preview_size))
            display_frame[10:10+preview_size, 10:10+ preview_size] = crop_preview
            cv2.rectangle(display_frame, (10, 10),
                          (10+preview_size, 10+preview_size), (255, 255, 255), 1)

        else:
            # No object detected
            cv2.putText(display_frame, "No object detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display FPS
        current_time = time.time()
        if current_time - last_fps_time >= fps_report_interval:
            fps = frame_count / fps_report_interval
            fps_text = f"FPS: {fps:1f}"
            last_fps_time = current_time
            frame_count = 0
        else:
            fps_text = ""
        
        # Display database info
        prototypes, labels = memory.get_all_prototypes()
        db_info = f"Objects in DB: {len(labels)}"

        # Put text on frame
        y_offset = 30
        cv2.putText(display_frame, db_info, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        if fps_text:
            cv2.putText(display_frame, fps_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Live Recognition", display_frame)

        # Keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\n"
                  "Switching to registration mode...")
            cap.release()
            cv2.destroyAllWindows()
            # Registration
            print(" Registration here...")
            break
        elif key == ord('z'):
            memory.clear()
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n"
          " Recognition stopped.")
    
def main():
    print("\n"
          "-" * 50)
    print("Personal Object Recognition")
    print("-" * 50)

    # Check if database has objects
    memory = MemoryManager()
    prototypes, labels = memory.get_all_prototypes()

    if len(labels) == 0:
        print("\n"
              " No objects in database.")
        print(" Please register objects first:")
        print("   python scripts/register.py")
        return
    
    print("\n"
          f" Database has {len(labels)} object(s):")
    for label in labels:
        print(f"  - {label}")

    print("\n"
          " Starting recognition in 3 seconds...")
    time.sleep(3)

    live_recognition()

# Test
if __name__ == '__main__':
    main()
