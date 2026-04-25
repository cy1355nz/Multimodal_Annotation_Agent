"""
File handling utilities for reading and processing files.
"""
import os
from utils.logger_handler import logger


def read_text_file(filepath: str) -> str:
    """
    Read text file content.

    Args:
        filepath: Path to the text file.

    Returns:
        Content of the text file as string.
    """
    try:
        if not os.path.exists(filepath):
            logger.error(f"[read_text_file] File {filepath} does not exist")
            return ""

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        logger.info(f"[read_text_file] Successfully read file: {filepath}, length: {len(content)}")
        return content
    except Exception as e:
        logger.error(f"[read_text_file] Failed to read file: {str(e)}")
        return ""


def list_files_with_extension(path: str, extensions: tuple) -> list:
    """
    List files in directory with specified extensions.

    Args:
        path: Directory path.
        extensions: Tuple of allowed file extensions.

    Returns:
        List of file paths.
    """
    files = []

    if not os.path.isdir(path):
        logger.error(f"[list_files_with_extension] {path} is not a directory")
        return files

    for filename in os.listdir(path):
        if filename.lower().endswith(extensions):
            files.append(os.path.join(path, filename))

    return files
