import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    
    def __init__(self):
        # Auto-detect hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use FP16 on GPU to save VRAM, fallback to FP32 on CPU
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        self.processor = None
        self.model = None
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, force_reload=False):
        if self.model is not None and not force_reload:
            return
            
        logger.info(f"Loading Depth Anything V2 on {self.device} with precision {self.dtype}...")
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully into VRAM/RAM.")

    def get_edsr_model_path(self):
        model_path = os.path.join(os.path.dirname(__file__), "EDSR_x4.pb")
        if not os.path.exists(model_path):
            logger.info("Downloading EDSR_x4 Super Resolution Model (~38MB)...")
            url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb"
            urllib.request.urlretrieve(url, model_path)
            logger.info("EDSR_x4 downloaded successfully.")
        return model_path

    def get_model(self):
        if self.model is None:
            self.load_model()
        return self.processor, self.model

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded. VRAM cleared.")
