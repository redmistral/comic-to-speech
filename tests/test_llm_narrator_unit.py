"""
Unit tests for the LLM-based comic narrator.

Tests the ComicNarrator class which uses OpenAI GPT-4 Vision to generate
audiobook-style narration from comic images.

Tested functionality:
- Initialization (requires API key)
- Base64 image encoding (used for API requests)
- Prompt generation (with/without panel context)
- Successful narration with mocked OpenAI responses
- Error handling when OpenAI API fails (rate limits, network errors)

All OpenAI API calls are mocked to avoid costs and ensure fast, deterministic tests.
"""
from unittest.mock import patch, MagicMock
import pytest
import base64
from narration.llm_narrator import ComicNarrator, narrate_panel


def test_narrator_init_without_api_key():
    """Verifies ComicNarrator initialization fails without API key"""
    with patch.dict('os.environ', {}, clear=True):
        # Remove OPENAI_API_KEY from environment
        with pytest.raises(ValueError) as exc_info:
            ComicNarrator(api_key=None)

        assert "api key" in str(exc_info.value).lower()


def test_narrator_encode_image_to_base64():
    """Verifies image base64 encoding produces correct output"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        narrator = ComicNarrator(api_key='test-key')

        test_bytes = b"test image data"
        result = narrator._encode_image_to_base64(test_bytes)

        assert isinstance(result, str)
        assert result == base64.b64encode(test_bytes).decode('utf-8')

        decoded = base64.b64decode(result)
        assert decoded == test_bytes


def test_narrator_create_prompt_with_panel_context():
    """Verifies prompt generation includes panel context when provided"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        narrator = ComicNarrator(api_key='test-key')

        prompt_with_context = narrator._create_narration_prompt(panel_number=2, total_panels=5)
        assert "panel 2 of 5" in prompt_with_context.lower()

        prompt_without_context = narrator._create_narration_prompt()
        assert "panel 2 of 5" not in prompt_without_context.lower()
        assert "this is panel" not in prompt_without_context.lower()


@patch("narration.llm_narrator.OpenAI")
def test_narrator_narrate_panel_success(mock_openai):
    """Verifies successful narration with mocked OpenAI API"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "A dramatic scene unfolds."
        mock_response.usage.total_tokens = 45
        mock_client.chat.completions.create.return_value = mock_response

        narrator = ComicNarrator(api_key='test-key')
        result = narrator.narrate_panel(b"fake_image", panel_number=1, total_panels=3)

        assert result["success"] is True
        assert result["narration"] == "A dramatic scene unfolds."
        assert result["tokens_used"] == 45
        assert result["error"] is None

        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4o"


@patch("narration.llm_narrator.OpenAI")
def test_narrator_narrate_panel_api_error(mock_openai):
    """Verifies narration handles OpenAI API failures gracefully"""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API rate limit exceeded")

        narrator = ComicNarrator(api_key='test-key')
        result = narrator.narrate_panel(b"fake_image")

        assert result["success"] is False
        assert result["narration"] == ""
        assert "rate limit" in result["error"].lower()
        assert result["tokens_used"] is None
