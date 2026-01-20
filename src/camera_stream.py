import numpy as np
import cv2

class CameraStream:
    """Handles camera initialization and frame acquisition."""
    def __init__(self, camera_idx: int = 0):
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            raise Exception(f"Error: No camera found in index [{camera_idx}].")
        
    def get_frame(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise Exception(f"Error: Could not read frame from camera.")
        return frame
    
    def release_destroy(self) -> None:
        self.cap.release()
        cv2.destroyAllWindows()
        