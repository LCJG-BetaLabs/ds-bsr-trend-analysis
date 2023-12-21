import sys
import logging

configured = False


def _configure_logger():
    # global configured
    logger = logging.getLogger("bsr_trend")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    configured = True


def get_logger():
    if not configured:
        _configure_logger()
    return logging.getLogger("bsr_trend")
