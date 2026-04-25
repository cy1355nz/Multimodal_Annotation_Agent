"""
Logger handler module for unified logging configuration.
"""
import logging
from utils.path_tool import get_abs_path
import os
from datetime import datetime

# Root directory for log storage
LOG_ROOT = get_abs_path("logs")

# Ensure log directory exists
os.makedirs(LOG_ROOT, exist_ok=True)

# Log format configuration
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = "annotation_agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None,
) -> logging.Logger:
    """
    Get a logger instance with console and file handlers.

    Args:
        name: Logger name.
        console_level: Logging level for console output.
        file_level: Logging level for file output.
        log_file: Custom log file path.

    Returns:
        Configured logger instance.
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers
    if logger_instance.handlers:
        return logger_instance

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger_instance.addHandler(console_handler)

    # File handler
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger_instance.addHandler(file_handler)

    return logger_instance


# Quick logger instance
logger = get_logger()

if __name__ == '__main__':
    logger.info("Info log")
    logger.error("Error log")
    logger.warning("Warning log")
    logger.debug("Debug log")
