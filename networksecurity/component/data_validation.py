from networksecurity.constant import training_pipeline
from networksecurity.entity.artifact_entity import DataIngestionArtifacts , DataValidationArtifacts
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os , sys


class DataValidation:
    def __init__(self , data_ingestion_artifacts : DataIngestionArtifacts , data_validation_config : DataValidationConfig) -> None:
        try :
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_validation_config=  data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

        except Exception as e :
            raise NetworkSecurityException(e , sys) 
    
    @staticmethod
    def read_data(file_path : str) -> pd.DataFrame:
        try :
            return pd.read_csv(file_path)
        except Exception as e :
            raise NetworkSecurityException(e , sys)

    
    def initiate_data_validation(self) -> DataValidationArtifacts :
        try :
            train_file_path = self.data_ingestion_artifacts.trained_file_path 
            test_file_path = self.data_ingestion_artifacts.test_file_path

            logging.info("Reading data from train and test csv")

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path
            )
            logging.info("Reading data from train and test csv completed")

      

        except Exception as e :
            raise NetworkSecurityException(e , sys) 
