import gradio as gr
import cv2
import numpy as np

from typing import List

from src.utils.config import load_config
from src.object_detector import ObjectDetector
from src.feature_extractor import FeatureExtractor
from src.memory_manager import MemoryManager

# Initialize components (cached)
def init_components():
    """Initializes model components once."""
    config = load_config()
    detector = ObjectDetector()
    extractor = FeatureExtractor()
    memory = MemoryManager()
    return config, detector, extractor, memory

config, detector, extractor, memory = init_components()

# Helper functions
def register_object_from_webcam(webcam_feed, label, progress=gr.Progress()):
    """Registers object from webcam feed."""
    if not label or webcam_feed is None:
        return " Please provide a label and use webcam."
    
    try:
        # Convert webcam image to bgr
        cv_image = cv2.cvtColor(webcam_feed, cv2.COLOR_RGB2BGR)

        # Detect object
        box = detector.detect(cv_image)

        if box is None:
            return " No object detected. Please hold object clearly in frame.", []
        
        # Crop object
        crop = detector.crop_object(cv_image, box)
        
        # Convert back to RGB for display
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        return  f" Ready to capture! Box detected. Press [Spacebar] to capture.", [crop_rgb]
    
    except Exception as e:
        return f" Error: {str(e)}", []
    
def capture_frame(captured_frames, webcam_feed, label):
    """Captures one frame for registration"""
    if
    
def recognize_image(image):
    """Recognizes object in uploaded image."""
    if image is None:
        return " Please upload an image."
    
    try:
        # Convert image
        cv_image = preprocess_image_to_bgr(image)
        
        # Detect object
        box = detector.detect(cv_image)
        
        if box is None:
            return "No object detected in image", None
        
        # Crop and extract features
        crop = detector.crop_object(cv_image, box)
        embedding = extractor.extract(crop)
        
        # Compare with database
        prototypes, labels = memory.get_all_prototypes()

        if len(prototypes) == 0:
            return "No objects in database. Please register objects first.", None

        # Calculate similarities
        prototypes_arrays = np.array(prototypes)
        similarities = np.dot(prototypes_arrays, embedding)

        # Get best match
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_label = labels[best_idx]

        # Thresholds
        config = load_config()
        threshold_high = config['recognition']['similarity_threshold_high']
        
        if best_similarity >= threshold_high:
            result = f" {best_label} (confidence: {best_similarity:.2f})"
        else:
            result = f" Unknown object (closest: {best_label}, similarity: {best_similarity:.2f})"
        
        # Draw bounding box on image
        x1, y1, x2, y2 = box
        result_image = cv_image.copy()
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        return result, result_image

    except Exception as e:
        return f" Error: {str(e)}", None
    
def get_database_status():
    """Gets current database status."""
    prototypes, labels = memory.get_all_prototypes()

    if len(labels) == 0:
        return "Database is empty. Please register objects first."
    
    status = f" **Database Status:** {len(labels)} object(s) registered \
               \n\n"
    for i, label in enumerate(labels, 1):
        status += f"{i}. **{label}** \
                    \n"

    return status

def clear_database():
    """Clears all objects from database."""
    memory.clear()
    return " Database cleared successfully!"

# Create Gradio interface
with gr.Blocks(title="Personal Object Recognizer") as demo:
    gr.Markdown("# Personal Object Recognizer")
    gr.Markdown("Few-shot learning system that recognizes your personal objects from just 4 examples.")

    with gr.Tab(" Register Object"):
        gr.Markdown("### Register a New Object")
        gr.Markdown("Upload 4 images of your object from different angles (0º, 90º, 180º, 270º).")

        with gr.Row():
            with gr.Column():
                image_inputs = []
                for i in range(4):
                    image_inputs.append(
                        gr.Image(label=f"View {i+1}", type="numpy")
                    )
                label_input = gr.Textbox(label="Object Name", placeholder="e.g.: My Mug")
                register_btn = gr.Button(" Register Object", variant="primary")

            with gr.Column():
                output_text = gr.Markdown()
                db_status = gr.Markdown()

        register_btn.click(
            fn=register_object,
            inputs=[gr.List(image_inputs), label_input],
            outputs=[output_text]
        ).then(
            fn=get_database_status,
            inputs=[],
            outputs=[db_status]
        )
        
    with gr.Tab(" Recognize Object"):
        gr.Markdown("### Recognize Object in Image")