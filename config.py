"""
Configuration Management for XHaloPathAnalyzer

Centralized configuration with environment variable support.
"""

import os
import torch
from dotenv import load_dotenv
from pathlib import Path
import logging

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """
    Centralized configuration management for XHaloPathAnalyzer.
    
    Loads settings from environment variables and provides defaults.
    Validates configuration on initialization.
    """
    
    # Halo API Settings
    HALO_API_ENDPOINT = os.getenv("HALO_API_ENDPOINT", "")
    HALO_API_TOKEN = os.getenv("HALO_API_TOKEN", "")
    
    # Model Settings
    MEDSAM_CHECKPOINT = os.getenv("MEDSAM_CHECKPOINT", "./models/medsam_vit_b.pth")
    MODEL_TYPE = os.getenv("MODEL_TYPE", "vit_b")  # vit_b, vit_l, vit_h
    
    # Application Settings
    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "500"))
    TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Image Processing Settings
    DEFAULT_TARGET_SIZE = int(os.getenv("DEFAULT_TARGET_SIZE", "1024"))
    JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "95"))
    
    # Device Configuration (automatically detect CUDA)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GeoJSON Settings
    MIN_POLYGON_AREA = int(os.getenv("MIN_POLYGON_AREA", "100"))
    SIMPLIFY_TOLERANCE = float(os.getenv("SIMPLIFY_TOLERANCE", "1.0"))
    
    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.
        
        Raises:
            ValueError: If required settings are missing or invalid
        """
        errors = []
        
        # Check required settings
        if not cls.HALO_API_ENDPOINT:
            errors.append("HALO_API_ENDPOINT is required")
        if not cls.HALO_API_TOKEN:
            errors.append("HALO_API_TOKEN is required")
        
        # Check model checkpoint exists
        if not Path(cls.MEDSAM_CHECKPOINT).exists():
            logger.warning(f"MedSAM checkpoint not found at {cls.MEDSAM_CHECKPOINT}")
            
        # Validate numeric settings
        if cls.MAX_IMAGE_SIZE_MB <= 0:
            errors.append("MAX_IMAGE_SIZE_MB must be positive")
        
        if errors:
            raise ValueError("Configuration errors: " + "; ".join(errors))
        
        # Create directories if they don't exist
        Path(cls.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        Path("./models").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Configuration validated successfully")
        logger.info(f"Using device: {cls.DEVICE}")
        
    @classmethod
    def get_temp_path(cls, filename: str) -> Path:
        """Generate path for temporary file"""
        return Path(cls.TEMP_DIR) / filename
    
    @classmethod
    def log_config(cls):
        """Log current configuration (excluding sensitive data)"""
        logger.info("=== Configuration ===")
        logger.info(f"API Endpoint: {cls.HALO_API_ENDPOINT}")
        logger.info(f"Model Checkpoint: {cls.MEDSAM_CHECKPOINT}")
        logger.info(f"Device: {cls.DEVICE}")
        logger.info(f"Temp Directory: {cls.TEMP_DIR}")
        logger.info("====================")
