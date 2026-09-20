import logging
import os
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)



LOG_DIR = os.path.join(
    PROJECT_ROOT,
    "logs"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)




LOG_FILE = (
    f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
)

LOG_FILE_PATH = os.path.join(
    LOG_DIR,
    LOG_FILE
)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format=(
        "[ %(asctime)s ] "
        "%(lineno)d "
        "%(name)s - "
        "%(levelname)s - "
        "%(message)s"
    ),
    level=logging.INFO,
    force=True
)