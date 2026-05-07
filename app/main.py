import os
# Disable OneDNN/MKL on Windows before any paddle/paddleocr imports
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""ML Search Service - Main Application Entry Point."""
from fastapi import FastAPI, Request, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
import logging  # noqa: E402

from .config import settings  # noqa: E402
from .api.endpoints import router  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.SERVICE_NAME}...")
    logger.info(f"Main API URL: {settings.MAIN_API_URL}")
    logger.info(f"YOLO Model: {settings.YOLO_MODEL}")
    logger.info(f"CLIP Model: {settings.CLIP_MODEL}")
    logger.info(f"GPU Enabled: {settings.USE_GPU}")
    
    # Pre-load models on startup (optional, can be lazy-loaded)
    try:
        logger.info("Pre-loading models...")
        from ..pipeline.yolo_detector import YOLOPartDetector
        from ..pipeline.embedding import CLIPEmbedding
        
        # Initialize detector to download model weights
        YOLOPartDetector(settings.YOLO_MODEL, settings.YOLO_CONFIDENCE_THRESHOLD)
        logger.info(f"YOLO model loaded: {settings.YOLO_MODEL}")
        
        # Initialize embedder to download CLIP model
        CLIPEmbedding(settings.CLIP_MODEL)
        logger.info(f"CLIP model loaded: {settings.CLIP_MODEL}")
        
    except Exception as e:
        logger.warning(f"Could not pre-load models: {e}")
        logger.info("Models will be loaded on first request.")
    
    logger.info(f"{settings.SERVICE_NAME} is ready!")
    
    yield
    
    # Shutdown
    logger.info(f"Shutting down {settings.SERVICE_NAME}...")


app = FastAPI(
    title="ML Search Service",
    description="Machine Learning service for image-based product search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
# Parse comma-separated origins, default to ["*"] if not set
_origins = settings.CORS_ORIGINS.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key middleware — rejects requests missing the correct key
# Skips check if API_KEY is not configured (open mode for local dev)
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    if settings.API_KEY:
        # /health and / are always public
        if request.url.path not in ("/health", "/"):
            key = request.headers.get("X-API-Key")
            if key != settings.API_KEY:
                raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return await call_next(request)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
def root():
    """Root endpoint - service health check."""
    return {
        "service": settings.SERVICE_NAME,
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
