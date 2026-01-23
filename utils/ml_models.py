"""
ML Model Integration

Wrapper for MedSAM and other segmentation models.
"""

import torch
import numpy as np
from segment_anything import sam_model_registry
from typing import Tuple, Optional
import logging
from config import is_mps_available

logger = logging.getLogger(__name__)


def _safe_load_state_dict(path: str):
    """
    Safely load model state dict from checkpoint file.
    
    Handles various PyTorch versions and checkpoint formats:
    - Always loads to CPU first to avoid CUDA-deserialize errors
    - Tries weights_only=True for newer PyTorch versions
    - Falls back for older PyTorch versions without weights_only parameter
    - Handles common checkpoint wrapper formats (state_dict, model, model_state_dict)
    
    Args:
        path: Path to checkpoint file
        
    Returns:
        State dict ready to load into model
    """
    # Always load to CPU first to avoid CUDA-deserialize errors.
    load_kwargs = {"map_location": torch.device("cpu")}

    # Try weights_only=True when supported (newer PyTorch). Fallbacks keep it compatible.
    try:
        sd = torch.load(path, **load_kwargs, weights_only=True)
    except TypeError:
        # Older torch doesn't have weights_only parameter
        sd = torch.load(path, **load_kwargs)
    except Exception:
        # If weights_only=True fails for some other reason (e.g., pickle error), fall back
        try:
            sd = torch.load(path, **load_kwargs, weights_only=False)
        except TypeError:
            # Very old PyTorch without weights_only parameter at all
            sd = torch.load(path, **load_kwargs)

    # Handle common checkpoint wrappers
    if isinstance(sd, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break

    return sd


class MedSAMPredictor:
    """
    Wrapper for MedSAM model inference.
    
    MedSAM is a medical image segmentation model based on SAM (Segment Anything Model).
    """
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = "cuda"):
        """
        Initialize MedSAM predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            model_type: Model architecture (vit_b, vit_l, vit_h)
            device: Device to run inference on (cuda, mps, or cpu)
        """
        # Respect user's device choice, only fall back if requested device unavailable
        valid_devices = ["cuda", "mps", "cpu"]
        if device not in valid_devices:
            logger.warning(f"Invalid device '{device}' specified. Valid devices: {valid_devices}. Falling back to CPU")
            self.device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            # Check for MPS as fallback
            if is_mps_available():
                logger.warning("CUDA requested but not available, falling back to MPS")
                self.device = "mps"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = "cpu"
        elif device == "mps" and not is_mps_available():
            logger.warning("MPS requested but not available, falling back to CPU")
            self.device = "cpu"
        else:
            self.device = device
        self.model_type = model_type
        
        try:
            self.model = self._load_model(checkpoint_path, model_type)
            self.model.eval()
            logger.info(f"Loaded MedSAM model ({model_type}) on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_model(self, checkpoint_path: str, model_type: str):
        """
        Load MedSAM model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model_type: Model architecture
            
        Returns:
            Loaded model
        """
        try:
            # Use the device that was already determined and validated in __init__
            device = torch.device(self.device)

            # IMPORTANT: build WITHOUT checkpoint to avoid internal torch.load(...)
            model = sam_model_registry[model_type](checkpoint=None)

            state_dict = _safe_load_state_dict(checkpoint_path)
            model.load_state_dict(state_dict)

            model.to(device)
            
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    @torch.no_grad()
    def predict(self, 
                image: np.ndarray, 
                point_coords: Optional[np.ndarray] = None,
                point_labels: Optional[np.ndarray] = None,
                box: Optional[np.ndarray] = None,
                multimask_output: bool = False) -> np.ndarray:
        """
        Run inference on image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            point_coords: Point prompts (N, 2) in (x, y) format
            point_labels: Labels for points (1 = foreground, 0 = background)
            box: Bounding box prompt (x1, y1, x2, y2)
            multimask_output: Whether to return multiple mask predictions
            
        Returns:
            Binary mask as numpy array
        """
        # Prepare image - convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Encode image
        with torch.no_grad():
            # Get image embedding
            image_embedding = self.model.image_encoder(image_tensor)
        
        # Prepare prompts
        prompt_points = None
        prompt_labels = None
        prompt_box = None
        
        if point_coords is not None and point_labels is not None:
            prompt_points = torch.from_numpy(point_coords).float().unsqueeze(0).to(self.device)
            prompt_labels = torch.from_numpy(point_labels).float().unsqueeze(0).to(self.device)
        
        if box is not None:
            prompt_box = torch.from_numpy(box).float().unsqueeze(0).to(self.device)
        
        # If no prompts provided, use automatic mode (center point)
        if prompt_points is None and prompt_box is None:
            h, w = image.shape[:2]
            # Use center point as default prompt
            center_point = np.array([[w // 2, h // 2]])
            prompt_points = torch.from_numpy(center_point).float().unsqueeze(0).to(self.device)
            prompt_labels = torch.ones((1, 1)).float().to(self.device)
        
        # Decode mask
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(prompt_points, prompt_labels) if prompt_points is not None else None,
                boxes=prompt_box,
                masks=None,
            )
            
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
        
        # Upscale masks to original image size
        masks = torch.nn.functional.interpolate(
            low_res_masks,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        
        # Get best mask based on IOU score
        if multimask_output:
            best_mask_idx = iou_predictions.argmax()
            mask = masks[0, best_mask_idx].cpu().numpy()
        else:
            mask = masks[0, 0].cpu().numpy()
        
        # Convert to binary mask
        binary_mask = mask > 0.0
        
        logger.info(f"Generated mask with {np.sum(binary_mask)} positive pixels")
        
        return binary_mask
    
    def predict_with_box(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Convenience method for box-based segmentation.
        
        Args:
            image: RGB image (H, W, 3)
            box: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Binary mask
        """
        return self.predict(image, box=box)
    
    def predict_with_points(self, image: np.ndarray, 
                          points: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Convenience method for point-based segmentation.
        
        Args:
            image: RGB image (H, W, 3)
            points: Point coordinates (N, 2)
            labels: Point labels (N,) - 1 for foreground, 0 for background
            
        Returns:
            Binary mask
        """
        return self.predict(image, point_coords=points, point_labels=labels)
    
    def batch_predict(self, images: list, prompts: list) -> list:
        """
        Run batch inference on multiple images.
        
        Args:
            images: List of RGB images
            prompts: List of prompt dictionaries
            
        Returns:
            List of binary masks
        """
        masks = []
        for image, prompt in zip(images, prompts):
            mask = self.predict(
                image,
                point_coords=prompt.get('points'),
                point_labels=prompt.get('labels'),
                box=prompt.get('box')
            )
            masks.append(mask)
        return masks
