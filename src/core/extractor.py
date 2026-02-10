import numpy as np
import cv2
import torch

from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from PIL import Image
from typing import List

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import set_seed

class FeatureExtractor:
    """
    Feature extractor using MobileNetV3-Small for object embeddings.
    
    Removes classification head, uses global average pooling 
    and returns L2-normalized feature vectors.
    """
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()
        
        # Set all random seeds for reproducibility
        set_seed()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading feature extractor on {self.device}.")

        # Load pretrained MobileNetV3-Small
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the classification head
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension (for MobileNetV3-Small, output of last conv is 576)
        self.embedding_dim = self.config['feature_extractor']['embedding_dim']
        
        # Define image preprocessing
        self.input_size = self.config['feature_extractor']['input_size']
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.logger.info(f"Feature extractor '{self.config['feature_extractor']['model_name']}' \
initialized - Embedding dim: {self.embedding_dim}")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts embedding from a single BGR image (OpenCV format).
        
        Args:
            image: BGR image array (H, W, C).
        
        Returns:
            Normalized embedding vector (L2 norm = 1).
        """
        # Convert BGR to RGB
        image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_ = Image.fromarray(image_)
        
        # Preprocess
        input_tensor = self.preprocess(image_).unsqueeze(0).to(self.device) # type: ignore

        with torch.no_grad(), torch.inference_mode():
            features = self.model(input_tensor)
            # Global average pooling
            embedding = torch.mean(features, dim=[2, 3]).squeeze()
        
        # L2 normalize
        embedding = embedding.cpu().numpy()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)
            self.logger.warning("Zero-norm embedding detected.")

        return embedding
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extracts embeddings from a batch of images.
        
        Args:
            images: List of BGR image arrays.
            
        Returns:
            Array of normalized embeddings (n, embedding_dim).
        """
        if not images:
            return np.array([])

        # Preprocess all images
        input_tensors = []
        for img in images:
            image_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_ = Image.fromarray(image_)
            input_tensor = self.preprocess(image_)
            input_tensors.append(input_tensor)
            
        input_batch = torch.stack(input_tensors).to(self.device)

        with torch.no_grad(), torch.inference_mode():
            features = self.model(input_batch)
            # Global average pooling over spatial dimensions
            embeddings = torch.mean(features, dim=[2, 3])

        # L2 normalize each embedding
        embeddings = embeddings.cpu().numpy()
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        valid_norms = norms > 0
        embeddings = np.where(valid_norms, embeddings / norms, 0)

        if not np.all(valid_norms):
            self.logger.warning(f"Zero-norm embeddings detected: {np.sum(~valid_norms.flatten())}")

        return embeddings
    