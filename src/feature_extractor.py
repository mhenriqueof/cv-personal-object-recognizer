import torch

from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms

from src.utils.loads import load_config
from src.utils.setups import setup_logger
class FeatureExtractor:
    """Extracts feature embeddings from object crops using a pretrained CNN."""
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading feature extractor on {self.device}.")

        # Load pretrained MobileNetV3-Small
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the classification head (last two layers: avgpool and classifier)
        # We'll take features from before the global avgpool (last convolutional layer)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension (for MobileNetV3-Small, output of last conv is 576)
        self.embedding_dim = self.config['feature_extractor']['embedding_dim']
        
        # Define image preprocessing
        self.input_size = self.config['feature_extractor']['input_size']
        self.preprocess = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.logger.info(f"Feature extractor {self.config['extractor']['model_name']} initialized - \n\
            Embedding dim: {self.embedding_dim}")
        