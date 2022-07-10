# coding = utf-8

import logging
import os


class LogRecoder(object):
    def __init__(self, logger_name, log_file_path, std_handler_level, file_handler_level):
        self.logger = logging.getLogger(logger_name)
        self.std_handler = logging.StreamHandler()
        if not os.path.isfile(log_file_path):
            log_file_dir = log_file_path[0:log_file_path.rfind(os.sep)]
            if not os.path.isdir(log_file_dir):
                os.makedirs(log_file_dir)
            with open(log_file_path, mode=r"w", encoding="utf-8") as f_handler:
                f_handler.close()
        self.file_handler = logging.FileHandler(log_file_path)

        self.logger.setLevel(logging.DEBUG)
        self.std_handler.setLevel(std_handler_level)
        self.file_handler.setLevel(file_handler_level)

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        self.std_handler.setFormatter(formatter)
        self.file_handler.setFormatter(formatter)

        self.logger.addHandler(self.std_handler)
        self.logger.addHandler(self.file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
