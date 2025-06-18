import logging
from datetime import datetime
import os
import sys
from typing import Optional

class Log():
    """
    A class to handle logging.
    """
    # single case log
    _shared_file_handler = None
    log_time = ""

    def __init__(self, log_name: str, file_name: Optional[str] = None):
        """
        Initialize the logger with file and console handlers.
        """
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.INFO)
        os.makedirs("logs", exist_ok=True)
        Log.log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Initialize shared file handler if not already done
        if Log._shared_file_handler is None:
            if file_name is None:
                file_name = "shared"
            Log._shared_file_handler = logging.FileHandler(f"logs/{Log.log_time}_{file_name}.log")
            file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M:%S')
            Log._shared_file_handler.setFormatter(file_formatter)

        self.logger.addHandler(Log._shared_file_handler)

        # Stream handler (for terminal output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d %H:%M:%S')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)