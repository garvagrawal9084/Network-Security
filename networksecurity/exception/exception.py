import sys 
from networksecurity.logging import logger

class NetworkSecurityException(Exception) :
    def __init__(self, error_message , error_detail:sys) :
        self.error_message = error_message , 
        _ ,_  , exec_tb = error_detail.exc_info()

        self.lineno = exec_tb.tb_lineno
        self.file_name = exec_tb.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        return f"Error occured in python scipt name {self.file_number} line number {self.lineno} error message , {self.error_message}"

if __name__ == "__main__" :
    try :
        logger.logging.info("Enter the try blocl")
        a = 1 / 0
        print("This will not be published")
    except Exception as e:
        raise NetworkSecurityException(e , sys)
