import cv2
import numpy as np
import time

class FPSTracker:
    """
    Frames Per Second tracker and display utility.
    
    Calculates running FPS over time and displays it on frame with color coding
    (green=good, yellow=ok, red=poor).
    
    Args:
        update_interval: How often to recalculate FPS (seconds).
    """
    def __init__(self, update_interval: float = 0.5):
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = time.time()
        self.update_interval = update_interval
        
    def update(self) -> float:
        """
        Updates FPS counter - call every frame.
        
        Returns:
            Current FPS value
        """
        self.frame_count += 1
        current_time = time.time()

        if current_time - self.last_time >= self.update_interval:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
        return self.fps
    
    def show(self, frame: np.ndarray) -> np.ndarray:
        """
        Draws FPS counter on frame.
        
        Args:
            frame: Input image.
            
        Returns:
            Copy of frame with FPS overlay.
        """
        display_frame = frame.copy()

        fps_text = f"FPS: {self.fps:.1f}"
        h, w = display_frame.shape[:2]

        # Choose color based on FPS
        if self.fps >= 30:
            color = (0, 255, 0) # Green
        elif self.fps >= 15:
            color = (0, 255, 255) # Yellow
        else:
            color = (0, 0, 255) # Red
            
        # Calculate position (top-right corner)
        (text_width, text_height), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        
        # Position
        x = w - text_width
        y = text_height - 5
        
        # Draw backgrond rectangle
        cv2.rectangle(display_frame, 
                     (x - 5, y - text_height),
                     (x + text_width + 5, y),
                     (0, 0, 0), -1)
        
        # Draw FPS text
        cv2.putText(display_frame, fps_text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return display_frame
        