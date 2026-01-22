import cv2
import numpy as np

from src.utils.config import load_config
from src.utils.system_mode import SystemMode
from src.camera_stream import CameraStream
from src.utils.get_object_name import get_object_name

from scripts.recognize_object import recognize_object
from scripts.register_object import register_object

class PersonalObjectRecognizer:
    def __init__(self):
        self.config = load_config()
        self.camera = CameraStream()
        self.mode = SystemMode.RECOGNIZE
        
        self.new_object_name: str
    
    def run(self):
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
            
            # System/Layout modes
            if self.mode == SystemMode.REGISTER:
                display_frame, stop_register_mode = register_object(frame, key, self.new_object_name)
                if stop_register_mode:
                    self.mode = SystemMode.RECOGNIZE
            else:
                display_frame= recognize_object(frame)
                
            # Show application
            cv2.imshow("Personal Object Recognizer", display_frame)

app = PersonalObjectRecognizer()
app.run()
