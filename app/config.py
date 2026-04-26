"""ML Search Service Configuration."""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings for ML Search Service."""
    
    # Service
    SERVICE_NAME: str = "ml-search-service"
    SERVICE_PORT: int = 8001
    
    # Main API
    MAIN_API_URL: str = os.getenv("MAIN_API_URL", "")
    
    # Model paths
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "")  # nano for speed, use 's' or 'm' for accuracy
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "")
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "")
    
    # Confidence thresholds
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", ""))
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", ""))
    
    # FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "")
    EMBEDDING_DIMENSION: int = 768  # Large patch14 uses 768
    TEXT_EMBEDDING_DIMENSION: int = 1024  # BGE-M3 uses 1024
    
    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() == "true"
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
