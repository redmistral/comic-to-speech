"""
Unit tests for individual AI task processing functions.

Tests three core worker tasks in isolation:
- OCR text extraction (process_ocr_task)
- Translation (process_translation_task)
- Text-to-speech synthesis (process_tts_task)

Each task is tested for:
- Success path with valid inputs
- Error handling with invalid inputs (None, empty strings)
- Exception handling when underlying services fail
- Proper result structure (success/error dictionaries)

External dependencies are mocked to isolate task logic from actual API calls.
"""
import pytest
from unittest.mock import patch, MagicMock
from workers.tasks import (
    process_ocr_task,
    process_translation_task,
    process_tts_task,
)

@patch("workers.tasks.ComicOCR")
def test_ocr_task_success(mock_ocr):
    """Verifies OCR successfully extracts text from comic image"""
    instance = mock_ocr.return_value
    instance.extract_text.return_value = {
        "text": "Hello",
        "panel_count": 1,
        "bubble_count": 1,
        "text_blocks": [],
        "confidence": 0.95,
        "narration_mode": "ocr"
    }

    result = process_ocr_task(b"fakeimage")

    assert result["success"] is True
    assert result["extracted_text"] == "Hello"
    assert result["panel_count"] == 1
    assert result["bubble_count"] == 1
    assert result["confidence"] == 0.95


@patch("workers.tasks.ComicOCR")
def test_ocr_task_failure(mock_ocr):
    """Verifies OCR handles extraction failures gracefully"""
    instance = mock_ocr.return_value
    instance.extract_text.side_effect = Exception("OCR boom")

    result = process_ocr_task(b"fake")

    assert result["success"] is False
    assert "error" in result
    assert "OCR boom" in result["error"] or "OCR" in result["error"]


@patch("workers.tasks.is_translation_available", return_value=True)
@patch("workers.tasks.translate_text")
def test_translation_task_success(mock_trans, _):
    """Verifies successful text translation"""
    mock_trans.return_value = "Hallo wereld"

    result = process_translation_task("Hello world", src_lang="en", tgt_lang="nl")

    assert result["success"] is True
    assert result["translated_text"] == "Hallo wereld"
    assert result["original_text"] == "Hello world"
    assert result["src_lang"] == "en"
    assert result["tgt_lang"] == "nl"


@patch("workers.tasks.is_translation_available", return_value=False)
def test_translation_task_unavailable(_):
    """Verifies translation handles unavailable model gracefully"""
    result = process_translation_task("Hello")

    assert result["success"] is False
    assert "not available" in result["error"].lower()


def test_translation_task_empty_text():
    """Verifies translation rejects empty text input"""
    result = process_translation_task("")

    assert result["success"] is False
    assert "no text" in result["error"].lower()


def test_translation_task_none_text():
    """Verifies translation handles None text input"""
    result = process_translation_task(None)

    assert result["success"] is False
    assert "error" in result


@patch("workers.tasks.is_translation_available", return_value=True)
@patch("workers.tasks.translate_text")
def test_translation_task_exception_handling(mock_trans, _):
    """Verifies translation exceptions are caught and reported"""
    mock_trans.side_effect = RuntimeError("Translation service timeout")

    result = process_translation_task("Hello")

    assert result["success"] is False
    assert "timeout" in result["error"].lower()
    assert result["original_text"] == "Hello"


@patch("workers.tasks.get_tts_client")
@patch("workers.tasks.AUDIO_DIR")
def test_tts_success(mock_audio_dir, mock_get_tts):
    """Verifies successful text-to-speech audio generation"""
    # Mock the TTS client
    client = MagicMock()
    mock_get_tts.return_value = client

    # Mock audio response
    mock_response = MagicMock()
    mock_response.audio_content = b"FAKE_AUDIO_DATA"
    client.synthesize_speech.return_value = mock_response

    # Mock AUDIO_DIR path operations
    mock_audio_path = MagicMock()
    mock_audio_dir.__truediv__ = MagicMock(return_value=mock_audio_path)

    result = process_tts_task("Hello")

    assert result["success"] is True
    assert "audio_id" in result
    assert "audio_url" in result
    assert result["characters_used"] == len("Hello")
    client.synthesize_speech.assert_called_once()


@patch("workers.tasks.get_tts_client")
def test_tts_failure(mock_get_tts):
    """Verifies TTS handles synthesis failures gracefully"""
    client = MagicMock()
    mock_get_tts.return_value = client

    client.synthesize_speech.side_effect = Exception("TTS failed")

    result = process_tts_task("Hello")

    assert result["success"] is False
    assert "error" in result
    assert "TTS" in result["error"] or "failed" in result["error"].lower()


def test_tts_empty_text():
    """Verifies TTS rejects empty text input"""
    result = process_tts_task("")

    assert result["success"] is False
    assert "no text" in result["error"].lower()


def test_tts_none_text():
    """Verifies TTS handles None text input"""
    result = process_tts_task(None)

    assert result["success"] is False
    assert "no text" in result["error"].lower()


@patch("workers.tasks.get_tts_client")
def test_tts_client_initialization_failure(mock_get_tts):
    """Verifies TTS handles client initialization failures"""
    mock_get_tts.side_effect = RuntimeError("Google credentials not found")

    result = process_tts_task("Hello world")

    assert result["success"] is False
    assert "not initialized" in result["error"].lower()
