import os
import sys
import numpy as np
import pandas as pd

"""
Defining common constant variable for training pipeline
"""

TARGET_COLUMN : str = "Result" , 
PIPELINE_NAME : str = "NetworkSecurity" 
ARTIFACT_DIR : str = "Artifacts" 
FILE_NAME : str = "phisingData.csv" 

TRAIN_FILE_NAME : str = "train.csv"
TEST_FILE_NAME : str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema" , "schema.yaml")


"""
DATA INGESTION related constant start with DATA_INTEGRATION VAR NAME
"""

DATA_INTEGRATION_COLLECTION_NAME: str = "NetworkData"
DATA_INTEGRATION_DATABASE_NAME: str = "networkSecuity"
DATA_INTEGRATION_DIR_NAME: str = "data_ingestion"
DATA_INTEGRATION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INTEGRATION_INGESTED_DIR: str = "ingested"
DATA_INTEGRATION_TRAIN_TEST_SPLIT_RATIO : float = 0.2


"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR  = "validated"
DATA_VALIDATION_INVALID_DIR  = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"




