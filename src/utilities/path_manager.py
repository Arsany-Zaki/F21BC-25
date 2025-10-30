import os
from pathlib import Path


def normalize_path(path_from_config: str) -> str:
    """Convert forward slashes from config to OS-appropriate path separators"""
    return os.path.normpath(path_from_config)


def ensure_directory_exists(directory_path: str) -> str:
    """Create directory if it doesn't exist and return normalized path"""
    normalized_path = normalize_path(directory_path)
    os.makedirs(normalized_path, exist_ok=True)
    return normalized_path


def join_paths(*paths) -> str:
    """Join multiple path components using OS-appropriate separators"""
    return os.path.join(*paths)