from networksecurity.entity.artifact_entity import DataIngestionArtifacts , DataValidationArtifacts
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file , write_yaml_file

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

    def validate_number_of_columns(self , dataframe : pd.DataFrame) -> bool :
        try :
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"The required number of columns : {number_of_columns}")
            logging.info(f"Data Frame has columns : {len(dataframe.columns)}") 

            if len(dataframe.columns) == number_of_columns :
                return True
            else :
                return False

        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def validate_numerical_columns_exist(self,  dataframe : pd.DataFrame) -> bool :
        try :
            numerical_columns = sum(dataframe[column].dtype == "int64" for column in dataframe.columns)
            numerical_columns_config = len(self._schema_config["numerical_columns"])

            logging.info(f"The required numerical columns : {numerical_columns_config}")
            logging.info(f"Dataframe numerical columns : {numerical_columns}")

            return numerical_columns_config == numerical_columns 

        except Exception as e :
            raise NetworkSecurityException(e , sys) 

    
    def detect_dataset_drift(self , base_df : pd.DataFrame , current_df : pd.DataFrame , threshold = 0.05) -> bool :
        try :
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1 , d2)

                if threshold <= is_same_dist.pvalue:
                    is_found = False 
                else :
                    is_found = True 
                    status = False
                report.update({column: {
                    "p_value" : float(is_same_dist.pvalue),
                    "drift_status" : is_found
                }})
            
            drift_report_dir_path = self.data_validation_config.drift_report_dir_path
            os.makedirs(drift_report_dir_path , exist_ok=True)

            drift_report_file_path = self.data_validation_config.drift_report_file_path 
            write_yaml_file(file_path=drift_report_file_path , content=report)

            return status

        except Exception as e :
            raise NetworkSecurityException(e , sys) 




    
    def initiate_data_validation(self) -> DataValidationArtifacts:
        try:

            train_file_path = self.data_ingestion_artifacts.trained_file_path
            test_file_path = self.data_ingestion_artifacts.test_file_path

            logging.info("Reading data from train and test csv")

            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            logging.info("Reading data from train and test csv completed")

            error_message = ""

            # Validate number of columns
            logging.info("Validate number of columns")

            train_column_status = self.validate_number_of_columns(
                dataframe=train_dataframe
            )

            if not train_column_status:
                error_message += "Training data does not contain all columns\n"

            test_column_status = self.validate_number_of_columns(
                dataframe=test_dataframe
            )

            if not test_column_status:
                error_message += "Test data does not contain all columns\n"

            logging.info("Validate number of columns end")

            # Validate numerical columns
            logging.info("Validate number of numerical columns")

            train_numerical_status = self.validate_numerical_columns_exist(
                train_dataframe
            )

            if not train_numerical_status:
                error_message += (
                    "Training data does not contain all numerical columns\n"
                )

            test_numerical_status = self.validate_numerical_columns_exist(
                test_dataframe
            )

            if not test_numerical_status:
                error_message += (
                    "Test data does not contain all numerical columns\n"
                )

            logging.info("Validate number of numerical columns end")

            # Detect data drift
            logging.info("Checking data drift")

            drift_status = self.detect_dataset_drift(
                base_df=train_dataframe,
                current_df=test_dataframe
            )

            # Final validation status
            status = (
                train_column_status
                and test_column_status
                and train_numerical_status
                and test_numerical_status
                and drift_status
            )

            # Create directory
            dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )

            os.makedirs(dir_path, exist_ok=True)

            # Save validated train data
            if status:
                train_dataframe.to_csv(
                    self.data_validation_config.valid_train_file_path,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.valid_test_file_path,
                    index=False,
                    header=True
                )

                valid_train_file_path = (
                    self.data_validation_config.valid_train_file_path
                )
                valid_test_file_path = (
                    self.data_validation_config.valid_test_file_path
                )

                invalid_train_file_path = None
                invalid_test_file_path = None

            else:
                train_dataframe.to_csv(
                    self.data_validation_config.invalid_train_file_path,
                    index=False,
                    header=True
                )

                test_dataframe.to_csv(
                    self.data_validation_config.invalid_test_file_path,
                    index=False,
                    header=True
                )

                valid_train_file_path = None
                valid_test_file_path = None

                invalid_train_file_path = (
                    self.data_validation_config.invalid_train_file_path
                )
                invalid_test_file_path = (
                    self.data_validation_config.invalid_test_file_path
                )

            # Create artifacts
            data_validation_artifacts = DataValidationArtifacts(
                validation_status=status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path=(
                    self.data_validation_config.drift_report_file_path
                )
            )

            logging.info("Data validation completed")

            return data_validation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)