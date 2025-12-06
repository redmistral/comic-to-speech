"""
Edge case and stress tests for the comic processing pipeline.

Tests unusual scenarios that may occur in production:
- Empty/blank images (OCR returns no text)
- Translation service unavailable (graceful degradation to original text)
- TTS quota exceeded (proper error reporting)
- Parallel pipeline execution (basic concurrency smoke test)

Includes one skipped integration test that uses actual OCR API with a black
image to verify behavior without mocks. This test is skipped by default to
avoid API costs and can be run manually when needed.

Note: The parallel execution test uses mocks, so it only verifies the
orchestration logic doesn't crash with threading. Real race conditions
(file conflicts, Redis contention) are not tested here.
"""
import pytest
from unittest.mock import patch
from workers.tasks import process_comic_full_pipeline, process_ocr_task

@patch("workers.tasks.process_ocr_task")
def test_empty_text_validation(mock_ocr):
    """Verifies pipeline fails gracefully when OCR returns empty text"""
    mock_ocr.return_value = {
        "success": True,
        "extracted_text": "",
        "panel_count": 0,
        "bubble_count": 0,
        "text_blocks": 0,
        "confidence": 0.0,
        "narration_mode": "ocr",
        "tokens_used": None
    }

    result = process_comic_full_pipeline(b"black_image_bytes")

    assert result["success"] is False
    assert "no text" in result["error"].lower() or "empty" in result["error"].lower()


@pytest.mark.skip(reason="Skipped by default - requires OCR API credentials")
def test_ocr_with_actual_black_image():
    """
    Verifies OCR behavior with completely black image (uses actual API, no mocks).

    Skipped because:
    - Requires valid OPENAI_API_KEY or Google Cloud credentials
    - Incurs API costs ($0.01+ per call)
    - Takes 2-5 seconds to execute

    To run manually: pytest tests/test_extreme_cases.py::test_ocr_with_actual_black_image -v
    """
    from PIL import Image
    import io

    img = Image.new('RGB', (100, 100), color='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    black_image_bytes = buf.getvalue()

    result = process_ocr_task(black_image_bytes)

    # Either success or failure is acceptable
    assert result["success"] is True or result["success"] is False

    if result["success"]:
        text = result.get("extracted_text", "")
        assert len(text) < 50, f"Black image should not extract much text, got: {text}"
        assert result.get("confidence", 0) < 0.5
    else:
        assert "error" in result

@patch("workers.tasks.process_ocr_task")
@patch("workers.tasks.process_tts_task")
def test_parallel_pipeline_execution_smoke_test(mock_tts, mock_ocr):
    """
    Verifies pipeline handles parallel execution without crashing.
    Note: Uses mocks, so doesn't catch real race conditions (file I/O, Redis, etc.)
    """
    import concurrent.futures

    mock_ocr.return_value = {
        "success": True,
        "extracted_text": "Hi",
        "panel_count": 1,
        "bubble_count": 1,
        "text_blocks": 0,
        "confidence": 0.99,
        "narration_mode": "ocr",
        "tokens_used": None
    }
    mock_tts.return_value = {
        "success": True,
        "audio_id": "test_audio_123",
        "audio_url": "/api/audio/test_audio_123",
        "characters_used": 2
    }

    def job():
        return process_comic_full_pipeline(b"fake_image_data")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: job(), range(5)))

    for r in results:
        assert r["success"] is True
        assert r["audio_id"] is not None


@patch("workers.tasks.process_ocr_task")
@patch("workers.tasks.process_translation_task")
@patch("workers.tasks.process_tts_task")
def test_translation_unavailable(mock_tts, mock_trans, mock_ocr):
    """Verifies pipeline continues with original text when translation is unavailable"""
    mock_ocr.return_value = {
        "success": True,
        "extracted_text": "Hello",
        "panel_count": 1,
        "bubble_count": 1,
        "text_blocks": 0,
        "confidence": 0.9,
        "narration_mode": "ocr",
        "tokens_used": None
    }
    mock_trans.return_value = {
        "success": False,
        "error": "Translation model not available"
    }
    mock_tts.return_value = {
        "success": True,
        "audio_id": "audio_fallback",
        "audio_url": "/api/audio/audio_fallback",
        "characters_used": 5
    }

    result = process_comic_full_pipeline(b"image_data", translate=True)

    assert result["success"] is True
    assert "translation_error" in result
    assert result["translation_error"] == "Translation model not available"

    mock_tts.assert_called_once()
    tts_call_args = mock_tts.call_args[0]
    assert tts_call_args[0] == "Hello"


@patch("workers.tasks.process_ocr_task")
@patch("workers.tasks.process_translation_task")
@patch("workers.tasks.process_tts_task")
def test_tts_quota(mock_tts, mock_trans, mock_ocr):
    """Verifies pipeline fails gracefully when TTS quota is exceeded"""
    mock_ocr.return_value = {
        "success": True,
        "extracted_text": "Text",
        "panel_count": 1,
        "bubble_count": 1,
        "text_blocks": 0,
        "confidence": 0.9,
        "narration_mode": "ocr",
        "tokens_used": None
    }
    mock_trans.return_value = {
        "success": True,
        "translated_text": "Tekst"
    }
    mock_tts.return_value = {
        "success": False,
        "error": "Quota exceeded"
    }

    result = process_comic_full_pipeline(b"image_data", translate=True)

    assert result["success"] is False
    assert "error" in result
    assert "quota" in result["error"].lower() or "exceeded" in result["error"].lower()
