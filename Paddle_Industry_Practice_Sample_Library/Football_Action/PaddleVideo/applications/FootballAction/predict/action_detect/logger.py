"""
logger
"""
import os
import logging

class Logger(logging.Logger):
    """Customized logger for news stripper
    """
    def __init__(self):
        super(Logger, self).__init__(self)
        if not os.path.exists('logs'):
            os.mkdir('logs')
        handler = logging.FileHandler("logs/action_detect.log")
        # handler.setLevel(logging.DEBUG)
        handler.setLevel(logging.INFO)

        format = "%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d %(message)s"
        datefmt = "%y-%m-%d %H:%M:%S"

        formatter = logging.Formatter(format, datefmt)
        handler.setFormatter(formatter)
        self.addHandler(handler)

