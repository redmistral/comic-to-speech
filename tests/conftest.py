"""
Common pytest configuration and fixtures for all tests
"""
import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path so we can import modules correctly
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "server"))

# Mock directory creation before importing any modules that try to create directories
original_mkdir = Path.mkdir

def mock_mkdir(self, mode=0o777, parents=False, exist_ok=False):
    """Mock mkdir that doesn't fail on /app/ paths"""
    try:
        original_mkdir(self, mode=mode, parents=parents, exist_ok=exist_ok)
    except (FileNotFoundError, PermissionError):
        if not exist_ok:
            raise
        # Silently pass if exist_ok=True and we can't create the directory
        pass

# Apply the patch globally for all tests
Path.mkdir = mock_mkdir
