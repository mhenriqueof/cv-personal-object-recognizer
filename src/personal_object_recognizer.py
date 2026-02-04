import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.get_object_name import get_object_name
from src.utils.system_mode import SystemMode
from src.utils.fps_tracker import FPSTracker

from src.models.camera_stream import CameraStream
from src.models.object_detector import ObjectDetector
from src.models.feature_extractor import FeatureExtractor
from src.models.memory_manager import MemoryManager

from src.recognize_object import RecognizeObject
from src.register_object import RegisterObject

class PersonalObjectRecognizer:
    def __init__(self):
        self.config = load_config()
        self.camera = CameraStream()
        self.detector = ObjectDetector()
        self.extractor = FeatureExtractor()
        self.memory = MemoryManager()
        self.fps_tracker = FPSTracker()
        
        self.rec = RecognizeObject(self.config, self.detector, self.extractor, self.memory)
        self.reg = RegisterObject(self.config, self.detector, self.extractor, self.memory)

        self.mode = SystemMode.RECOGNIZE
        self.new_object_name: str
    
    def run(self, show_fps: bool = False):
        while True:
            # Get camera frame
            frame = self.camera.get_frame()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.camera.release_destroy()
                break
            elif key == ord('r'):
                self.new_object_name = get_object_name()
                self.mode = SystemMode.REGISTER
            elif key == ord('c'):
                self.memory.clear()
            
            # System/Layout modes
            if self.mode == SystemMode.REGISTER:
                display_frame, self.mode = self.reg.register(frame, key, self.new_object_name)
            else:
                display_frame = self.rec.recognize(frame)
            
            if show_fps:
                # Update FPS
                self.fps_tracker.update()
                # Show FPS
                display_frame = self.fps_tracker.show(display_frame)
                
            # Show application
            cv2.imshow("Personal Object Recognizer", display_frame)
