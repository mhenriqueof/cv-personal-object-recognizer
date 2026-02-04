import cv2
import numpy as np
import time

class FPSTracker:
    """Simple FPS tracker and display utility."""
    def __init__(self, update_interval: float = 0.5):
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.update_interval = update_interval
        
    def update(self) -> float:
        """Updates every frame."""
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_time >= self.update_interval:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
        return self.fps
    
    def show(self, frame: np.ndarray) -> np.ndarray:
        """
        Shows FPS on frame.
        
        Args:
            frame: Image to show on.
            
            Args:
                frame: Image to show on.
        """
        display_frame = frame.copy()

        fps_text = f"FPS: {self.fps:.1f}"
        h, w = display_frame.shape[:2]

        # Choose color based on FPS
        if self.fps >= 30:
            color = (0, 255, 0) # green
        elif self.fps >= 15:
            color = (0, 255, 255) # yellow
        else:
            color = (0, 0, 255) # red
            
        # Calculate position
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        x = w - text_width
        y = text_height
        
        cv2.putText(display_frame, fps_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
        