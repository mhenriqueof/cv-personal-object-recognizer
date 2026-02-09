import numpy as np
import cv2

class CameraStream:
    """
    Simple wrapper for OpenCV camera stream.
    
    Handles camera initialization, frame capture and cleanup.
    
    Args:
        camera_idx: Camera index (default: 0 for default webcam).
        
    Raises:
        RuntimeError: If camera cannot be opened.
    """
    def __init__(self, camera_idx: int = 0):
        self.cap = cv2.VideoCapture(camera_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"No camera found at index [{camera_idx}].")
        
    def get_frame(self) -> np.ndarray:
        """
        Capture a single frame from the camera.
        
        Returns:
            BGR image as numpy array, or None if capture fails.
            
        Raises:
            RuntimeError: If frame cannot be read.
        """
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame from camera.")
        return frame
    
    def release_destroy(self) -> None:
        """Releases camera resource and closes OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()
        