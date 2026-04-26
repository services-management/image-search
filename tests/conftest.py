import sys
import pytest
from unittest.mock import MagicMock


def _ensure_mock_module(name):
    """Inject a mock module into sys.modules if not present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


@pytest.fixture(autouse=True)
def mock_ml_models(monkeypatch):
    """Automatically mock heavy ML models for all tests unless marked as integration."""

    # Ensure mock modules exist so monkeypatch works even when libs aren't installed
    _ensure_mock_module("ultralytics")
    _ensure_mock_module("transformers")
    _ensure_mock_module("transformers.CLIPModel")
    _ensure_mock_module("transformers.CLIPProcessor")
    _ensure_mock_module("sentence_transformers")
    _ensure_mock_module("paddleocr")

    # Mock YOLO
    mock_yolo = MagicMock()
    mock_yolo.return_value = []
    monkeypatch.setattr("ultralytics.YOLO", lambda x: mock_yolo)

    # Mock CLIP
    mock_clip = MagicMock()
    monkeypatch.setattr("transformers.CLIPModel.from_pretrained", lambda x: mock_clip)
    monkeypatch.setattr("transformers.CLIPProcessor.from_pretrained", lambda x: MagicMock())

    # Mock SentenceTransformers (BGE-M3)
    mock_st = MagicMock()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *args, **kwargs: mock_st)

    # Mock PaddleOCR
    mock_paddle = MagicMock()
    monkeypatch.setattr("paddleocr.PaddleOCR", lambda *args, **kwargs: mock_paddle)
