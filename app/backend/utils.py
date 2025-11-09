import logging

class Logger:
    def setup_logger():
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        return logger