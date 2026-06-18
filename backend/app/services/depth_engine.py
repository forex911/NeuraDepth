import torch
import numpy as np
import cv2
from PIL import Image
from .model_manager import ModelManager
import logging

logger = logging.getLogger(__name__)

class DepthEngine:
    def __init__(self):
        self.manager = ModelManager.get_instance()

    def generate_depth(self, pil_image: Image.Image) -> np.ndarray:
        """
        Stage 2 & 4: Generates dense depth map using Multi-Scale Fusion.
        """
        logger.info("Starting Multi-Scale Depth inference...")
        processor, model = self.manager.get_model()
        
        orig_w, orig_h = pil_image.size
        
        # We process the image at two scales: 512px for global stability, 1024px for local details
        scales = [512.0, 1024.0]
        depth_maps = []
        
        with torch.no_grad():
            for max_dim in scales:
                scale = max_dim / max(orig_w, orig_h)
                if scale < 1.0:
                    infer_img = pil_image.resize((int(orig_w * scale), int(orig_h * scale)), Image.Resampling.LANCZOS)
                else:
                    # If the image is smaller than the target scale, we still process it at its native resolution
                    infer_img = pil_image

                inputs = processor(images=infer_img, return_tensors="pt").to(self.manager.device)
                
                # Cast inputs to FP16 if model is in FP16
                if self.manager.dtype == torch.float16:
                    inputs["pixel_values"] = inputs["pixel_values"].half()

                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

                # Upscale the raw depth tensor back to the original HD resolution while still on GPU
                depth_tensor = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=(orig_h, orig_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                # Offload to CPU and cast to 32-bit float for rendering pipeline
                depth_map = depth_tensor.cpu().numpy().astype(np.float32)

                # Normalize 32-bit array to [0.0, 1.0] for this specific scale
                d_min = depth_map.min()
                d_max = depth_map.max()
                if d_max - d_min > 1e-6:
                    depth_map = (depth_map - d_min) / (d_max - d_min)
                else:
                    depth_map = np.zeros_like(depth_map)
                    
                depth_maps.append(depth_map)

        # Stage 4: Weighted Multi-Scale Fusion
        # Blend the global structure (512px) with the local details (1024px)
        logger.info("Fusing multiple depth scales...")
        fused_depth = (depth_maps[0] * 0.4) + (depth_maps[1] * 0.6)
        
        logger.info("Depth inference complete.")
        return fused_depth

    def refine_depth(self, depth_map: np.ndarray, original_cv2_image: np.ndarray) -> np.ndarray:
        """
        Stage 3: Edge-Aware Refinement (Guided Filtering)
        """
        logger.info("Refining depth map with original image edges...")
        gray = cv2.cvtColor(original_cv2_image, cv2.COLOR_BGR2GRAY)
        
        # Histogram equalization to enhance guide edges
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)

        try:
            # Snap the blurry AI depth gradient to the sharp pixel boundaries of the HD image
            refined = cv2.ximgproc.createGuidedFilter(guide=clahe, radius=8, eps=0.01).filter(depth_map)
        except AttributeError:
            # Fallback
            refined = cv2.bilateralFilter(depth_map, d=9, sigmaColor=0.1, sigmaSpace=15.0)

        # Re-normalize just in case
        d_min = refined.min()
        d_max = refined.max()
        if d_max - d_min > 1e-6:
            refined = (refined - d_min) / (d_max - d_min)
            
        return refined

    def recover_details(self, depth_map: np.ndarray, original_cv2_image: np.ndarray, strength=0.25) -> np.ndarray:
        """
        Stage 5: Detail Recovery using Laplacian High-Frequency Fusion
        Recovers micro-textures like hair, fabric folds, and architectural features.
        """
        logger.info("Recovering micro-textures and high-frequency details...")
        
        # Convert original to grayscale float32
        gray = cv2.cvtColor(original_cv2_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Extract high-frequency details using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        
        # Filter noise from the laplacian (we only want strong structural details, not sensor noise)
        laplacian = cv2.bilateralFilter(laplacian, d=5, sigmaColor=0.1, sigmaSpace=5.0)
        
        # Blend the high frequency details into the depth map
        fused = depth_map + (laplacian * strength)
        
        # Re-normalize
        d_min = fused.min()
        d_max = fused.max()
        if d_max - d_min > 1e-6:
            fused = (fused - d_min) / (d_max - d_min)
            
        return fused

    def upscale_depth(self, depth_map: np.ndarray, target_resolution: tuple) -> np.ndarray:
        """
        Stage 6: High-Res Upscaling using EDSR via OpenCV DNN Superres.
        """
        logger.info(f"Upscaling depth map to {target_resolution}...")
        try:
            from cv2 import dnn_superres
            sr = dnn_superres.DnnSuperResImpl_create()
            model_path = self.manager.get_edsr_model_path()
            sr.readModel(model_path)
            sr.setModel("edsr", 4)
            
            # Convert to 8-bit for OpenCV DNN Superres
            depth_8u = (depth_map * 255.0).astype(np.uint8)
            upscaled_8u = sr.upsample(depth_8u)
            
            # Resize to exact target resolution and restore 32-bit
            upscaled_8u = cv2.resize(upscaled_8u, target_resolution, interpolation=cv2.INTER_CUBIC)
            return upscaled_8u.astype(np.float32) / 255.0
        except Exception as e:
            logger.warning(f"AI Super Resolution failed or skipped, falling back to Lanczos: {e}")
            return cv2.resize(depth_map, target_resolution, interpolation=cv2.INTER_LANCZOS4)
