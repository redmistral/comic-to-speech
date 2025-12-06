"""
Unit tests for the Flask interface server validation logic.

Tests the file upload validation function (validate_image_upload) which enforces:
- File presence (rejects None/missing files)
- File type restrictions (only image formats: jpg, jpeg, png, gif, webp)
- File size limits (10MB maximum)

Includes boundary tests for the exact 10MB size limit to verify the validation
logic uses > operator (10MB passes, 10MB+1 fails).

Uses fake file objects to test validation without actual file I/O.
"""
import pytest
import sys
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from interface_server import validate_image_upload

def test_no_file_uploaded():
    """Verifies validation fails when no file is provided"""
    ok, error = validate_image_upload(None)
    assert ok is False
    assert "no file" in error.lower()

def test_wrong_file_type():
    """Verifies validation fails for non-image file types"""
    class FakeFile:
        filename = "test.txt"

    ok, error = validate_image_upload(FakeFile())
    assert ok is False
    assert "unsupported" in error.lower() or "type" in error.lower()


def test_file_too_large():
    """Verifies validation rejects files exceeding 10MB limit"""
    class FakeFile:
        filename = "huge_comic.jpg"

        def seek(self, position, whence=0):
            pass

        def tell(self):
            # Return size larger than 10MB limit
            return 11 * 1024 * 1024

    ok, error = validate_image_upload(FakeFile())
    assert ok is False
    assert "large" in error.lower() or "size" in error.lower()


def test_file_size_exact_boundary_conditions():
    """Verifies exact 10MB boundary conditions (10MB passes, 10MB+1 fails)"""
    MAX_SIZE = 10 * 1024 * 1024  # 10MB in bytes

    # Test cases: (size, should_pass, description)
    test_cases = [
        (MAX_SIZE - 1, True, "10MB - 1 byte should pass"),
        (MAX_SIZE, True, "Exactly 10MB should pass (using > not >=)"),
        (MAX_SIZE + 1, False, "10MB + 1 byte should fail"),
        (MAX_SIZE + 1024, False, "10MB + 1KB should fail"),
    ]

    for size, should_pass, description in test_cases:
        class FakeFile:
            filename = "test.jpg"

            def seek(self, position, whence=0):
                pass

            def tell(self):
                return size

        ok, error = validate_image_upload(FakeFile())
        assert ok == should_pass, f"{description} (size={size})"

        if should_pass:
            assert error is None
        else:
            assert "large" in error.lower() or "size" in error.lower()


def test_valid_image_file():
    """Verifies validation passes for valid image files"""
    class FakeFile:
        filename = "comic.png"

        def seek(self, position, whence=0):
            pass

        def tell(self):
            # Return size within 10MB limit
            return 5 * 1024 * 1024

    ok, error = validate_image_upload(FakeFile())
    assert ok is True
    assert error is None


def test_various_image_extensions():
    """Verifies all supported image extensions are accepted"""
    supported_extensions = ["comic.jpg", "comic.jpeg", "comic.png", "comic.gif", "comic.webp"]

    for filename in supported_extensions:
        class FakeFile:
            def __init__(self, name):
                self.filename = name

            def seek(self, position, whence=0):
                pass

            def tell(self):
                return 1024  # 1KB file

        ok, error = validate_image_upload(FakeFile(filename))
        assert ok is True, f"Extension {filename} should be valid"
