import logging
import sys


def setup_logger(name: str = "traffic_vision"):
    logger = logging.getLogger(name)

    # Prevent duplicate logs if initialized multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # Standardize the trace format for easy parsing later
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Stream to console (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# Expose a global logger instance
logger = setup_logger()