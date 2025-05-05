__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import logging
import os
import sys


def logger_initialization(
    log_path: str, logger: str, loglevel: str = "DEBUG", loglevel_msg: str = None
) -> logging.Logger:
    """
    Message Logger Initialization
    """

    # msg logger
    msg_logger = logging.getLogger(logger)

    if msg_logger.handlers:
        return msg_logger

    # Create directories
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    # create file handler (outputs to file)
    path_log = os.path.join(log_path, f"{logger}.log")
    file_handler = logging.FileHandler(path_log)

    # set logging levels
    msg_logger.setLevel(loglevel)
    file_handler.setLevel(loglevel)

    log_formatter = logging.Formatter(
        "%(levelname)-8s [%(asctime)s] (%(filename)s:%(lineno)s) --- %(message)s ", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(log_formatter)

    # create stream handler (prints to stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(loglevel_msg if loglevel_msg else loglevel)

    # create stream formatter
    stream_formatter = logging.Formatter(f"%(levelname)-2s {logger}: [%(filename)s:%(lineno)s]:      %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # add handlers
    msg_logger.addHandler(file_handler)
    msg_logger.addHandler(stream_handler)
    msg_logger.propagate = False

    return msg_logger
