import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.system_mode import SystemMode
from src.camera_stream import CameraStream

from scripts.recognize_object import recognize_object
from scripts import register_object

class PersonalObjectRecognizer:
    def __init__(self):
        self.config = load_config()
        self.camera = CameraStream()
        self.mode = SystemMode.RECOGNIZE
    
    def run(self):
        while True:
            # Get camera frame
            frame = self.camera.get_frame()
            
            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.camera.release_destroy()
                break
            elif key == ord('c'):
                self.mode = SystemMode.REGISTER
            
            # System/Layout modes
            if self.mode == SystemMode.RECOGNIZE:
                display_frame = recognize_object(frame)
            elif self.mode == SystemMode.REGISTER:
                display_frame = register_object(frame)
            
            # Show application
            cv2.imshow("Personal Object Recognizer", display_frame)

app = PersonalObjectRecognizer()
app.run()
