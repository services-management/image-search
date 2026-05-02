"""ML Search Service Configuration."""
import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings for ML Search Service."""
    
    # Service
    SERVICE_NAME: str = os.getenv("SERVICE_NAME", "")
    SERVICE_PORT: int = int(os.getenv("SERVICE_PORT", ""))
    
    # Main API
    MAIN_API_URL: str = os.getenv("MAIN_API_URL", "")
    
    # Model paths
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "")  
    CLIP_MODEL: str = os.getenv("CLIP_MODEL", "")
    
    # Confidence thresholds
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", ""))
    OCR_CONFIDENCE_THRESHOLD: float = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", ""))
    
    # FAISS
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    
    # Text embedding (BGE-M3)
    TEXT_MODEL: str = os.getenv("TEXT_MODEL", "BAAI/bge-m3")
    TEXT_EMBEDDING_DIMENSION: int = int(os.getenv("TEXT_EMBEDDING_DIMENSION", "1024"))
    
    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "").lower() == "true"
    
    # CORS
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
