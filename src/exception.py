import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Generate a detailed error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "error occurred in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    logging.error(error_message)
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)
        
        logging.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message