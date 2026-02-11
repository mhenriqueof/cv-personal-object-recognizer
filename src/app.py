"""
Main Application Class

Orchestrates the complete object recognition pipeline.
"""

import cv2

from src.utils.config import load_config
from src.utils.system_mode import SystemMode
from src.utils.fps_tracker import FPSTracker

from src.core.camera import CameraStream
from src.core.detector import ObjectDetector
from src.core.extractor import FeatureExtractor
from src.core.memory import MemoryManager

from src.recognizer import RecognizeObject
from src.register import RegisterObject

class PersonalObjectRecognizer:
    """
    Main application orchestrator for Personal Object Recognizer.
    
    Orchestrates the complete object recognition pipeline.
    
    Attributes:
        config: Application configurations.
        camera: Camera stream handler.
        detector: Object detection (YOLO).
        extractor: Feature extraction (MobileNetV3).
        memory: Objects storage manager.
        recognize: Recognition module.
        register: Registration module.
        system_mode: Current system mode (Recognize/Register).
    """
    def __init__(self):
        self.config = load_config()
        self.fps_tracker = FPSTracker()
        
        self.camera = CameraStream()
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.memory = MemoryManager()
        
        self.recognize = RecognizeObject(self.config, self.detector, self.extractor, self.memory)
        self.register = RegisterObject(self.config, self.detector, self.extractor, self.memory)

        self.system_mode = SystemMode.RECOGNIZE
    
    def run(self, show_fps: bool = False) -> None:
        """
        Runs the main application loop.
        
        Args:
            show_fps: Whether to display FPS counter on screen.
        """
        while True:
            # Get camera frame
            frame = self.camera.get_frame()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if (key == ord('q')) and (self.system_mode is SystemMode.RECOGNIZE):
                self.camera.release_destroy()
                break
                
            # System mode
            if self.system_mode == SystemMode.RECOGNIZE:
                display_frame, self.system_mode = self.recognize.process(frame, key)
            else:
                display_frame, self.system_mode = self.register.process(frame, key)
            
            # Display FPS
            if show_fps:
                # Update FPS
                self.fps_tracker.update()
                # Show FPS
                display_frame = self.fps_tracker.show(display_frame)
                
            # Show application
            cv2.imshow("Personal Object Recognizer", display_frame)
