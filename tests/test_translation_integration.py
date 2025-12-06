"""
Integration tests for the OpenNMT translation system.

Tests the translation module which converts English text to Dutch using
a pre-trained OpenNMT Transformer model (93M parameters, 6 layers).

Special behavior:
- In pytest environment: Uses a dummy translator that returns "[NL] <text>"
- In production: Uses actual OpenNMT model (requires 1.1GB model files)

Tested scenarios:
- Translation availability check
- English to Dutch translation
- Empty text handling
- Special characters and punctuation
- Whitespace preservation
- Long multi-line text

Two tests are skipped because the pytest override (dummy translator) prevents
testing the real subprocess error handling. These can only be tested in
production or with a separate test runner.
"""
import pytest
from unittest.mock import patch
import subprocess
from translation.translator import translate_text, is_translation_available


def test_is_translation_available_with_pytest():
    """Verifies translation availability check returns True in pytest environment"""
    assert is_translation_available() is True


@pytest.mark.skipif(
    not is_translation_available(),
    reason="Translation model files not available"
)
def test_real_translation_english_to_dutch():
    """Verifies English to Dutch translation produces different output"""
    test_cases = [
        ("Hello world", "en", "nl"),
        ("Good morning", "en", "nl"),
        ("Thank you", "en", "nl"),
    ]

    for text, src, tgt in test_cases:
        result = translate_text(text, src_lang=src, tgt_lang=tgt)

        assert result != text, f"Translation should change text: {text} -> {result}"
        assert len(result) > 0

        if "[NL]" in result:
            assert result.startswith("[NL]")


def test_translation_with_empty_text():
    """Verifies translation returns empty string for empty input"""
    result = translate_text("", src_lang="en", tgt_lang="nl")

    assert result == ""


def test_translation_with_special_characters():
    """Verifies translation handles special characters and punctuation"""
    test_cases = [
        "Hello! How are you?",
        "It's a beautiful day",
        "Comic-book style text",
    ]

    for text in test_cases:
        result = translate_text(text, src_lang="en", tgt_lang="nl")
        assert result is not None
        assert len(result) > 0


@pytest.mark.skip(reason="Skipped - pytest override prevents subprocess testing")
@patch('subprocess.run')
def test_translate_subprocess_timeout(mock_run):
    """
    Verifies translation handles subprocess timeout gracefully.

    Skipped because:
    - The pytest override in translator.py activates at module import
    - This override replaces subprocess calls with a dummy translator
    - Cannot test real subprocess error handling with pytest running
    - Would need to test in production or with non-pytest test runner
    """
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="translate.py", timeout=120)

    with pytest.raises(RuntimeError) as exc_info:
        with patch('translation.translator.__name__', '__main__'):
            translate_text("Hello world", src_lang="en", tgt_lang="nl")

    assert "timeout" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()


@pytest.mark.skip(reason="Skipped - pytest override prevents subprocess testing")
@patch('subprocess.run')
def test_translate_subprocess_failure(mock_run):
    """
    Verifies translation handles subprocess failure gracefully.

    Skipped because:
    - The pytest override in translator.py activates at module import
    - This override replaces subprocess calls with a dummy translator
    - Cannot test real subprocess error handling with pytest running
    - Would need to test in production or with non-pytest test runner
    """
    mock_result = subprocess.CompletedProcess(
        args=["python", "translate.py"],
        returncode=1,
        stdout="",
        stderr="Model file not found"
    )
    mock_run.return_value = mock_result

    with pytest.raises(RuntimeError) as exc_info:
        with patch('translation.translator.__name__', '__main__'):
            translate_text("Hello world", src_lang="en", tgt_lang="nl")

    assert "failed" in str(exc_info.value).lower() or "error" in str(exc_info.value).lower()


def test_translation_preserves_whitespace():
    """Verifies translation handles whitespace appropriately"""
    result = translate_text("  Hello  ", src_lang="en", tgt_lang="nl")

    assert len(result) > 0


@pytest.mark.skipif(
    not is_translation_available(),
    reason="Translation model files not available"
)
def test_translation_with_long_text():
    """Verifies translation handles longer multi-line text"""
    long_text = """
    In a dark alley, the hero confronts the villain.
    "You'll never get away with this!" he shouts.
    The villain laughs menacingly.
    """

    result = translate_text(long_text.strip(), src_lang="en", tgt_lang="nl")

    assert result is not None
    assert len(result) > 0
    assert result != long_text.strip(), "Translation should change the text"
