from datetime import datetime
import os

from networksecurity.constant import training_pipeline


class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = None) -> None:
        
        if timestamp is None:
            timestamp = datetime.now()

        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR

        self.artifact_dir = os.path.join(
            self.artifact_name,
            timestamp_str
        )

        self.timestamp = timestamp

class DataIngestionConfig :
    def __init__(self , training_pipeline_config : TrainingPipelineConfig) -> None:

        self.data_ingestion_dir : str = os.path.join(
            training_pipeline_config.artifact_dir ,
            training_pipeline.DATA_INTEGRATION_DIR_NAME ,
        )

        self.feature_store_file_path : str = os.path.join(
            self.data_ingestion_dir , 
            training_pipeline.DATA_INTEGRATION_FEATURE_STORE_DIR
        )

        self.training_file_path : str = os.path.join(
            self.data_ingestion_dir , 
            training_pipeline.DATA_INTEGRATION_INGESTED_DIR , 
            training_pipeline.TRAIN_FILE_NAME 
        )

        self.test_file_path : str = os.path.join(
            self.data_ingestion_dir , 
            training_pipeline.DATA_INTEGRATION_INGESTED_DIR , 
            training_pipeline.TEST_FILE_NAME 
        )

        self.train_test_split_raio : float = training_pipeline.DATA_INTEGRATION_TRAIN_TEST_SPLIT_RATIO 

        self.collection_name : str = training_pipeline.DATA_INTEGRATION_COLLECTION_NAME 

        self.database_name : str = training_pipeline.DATA_INTEGRATION_DATABASE_NAME


class DataValidationConfig :
    def __init__(self , training_pipeline_config : TrainingPipelineConfig) -> None:

        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , training_pipeline.DATA_VALIDATION_DIR_NAME)

        self.valid_data_dir = os.path.join(self.data_validation_dir , training_pipeline.DATA_VALIDATION_VALID_DIR)

        self.invalid_data_dir = os.path.join(self.data_validation_dir , training_pipeline.DATA_VALIDATION_INVALID_DIR) ; 

        self.valid_train_file_path = os.path.join(self.valid_data_dir , training_pipeline.TRAIN_FILE_NAME)

        self.valid_test_file_path = os.path.join(self.valid_data_dir , training_pipeline.TEST_FILE_NAME)

        self.invalid_train_file_path = os.path.join(self.invalid_data_dir , training_pipeline.TRAIN_FILE_NAME) 

        self.invalid_test_file_path = os.path.join(self.invalid_data_dir , training_pipeline.TEST_FILE_NAME)

        self.drift_report_dir_path = os.path.join(self.data_validation_dir , training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR)

        self.drift_report_file_path = os.path.join(self.drift_report_dir_path , training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)