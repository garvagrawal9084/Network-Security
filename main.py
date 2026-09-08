from networksecurity.component.data_ingestion import DataIngestion
from networksecurity.component.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try :
        training_pipeline_config = TrainingPipelineConfig() 

        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config) 

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config) 
        logging.info("Initiate the data ingestion")

        data_ingestion_artifacts = data_ingestion.inititate_data_ingestion()
        logging.info("Data ingestion completed")
        
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        
        data_validation = DataValidation(data_ingestion_artifacts=data_ingestion_artifacts , data_validation_config=data_validation_config)
        logging.info("Data validation start")
        data_validation_artifacts = data_validation.initiate_data_validation() 
        logging.info("Data validation ended")

    except Exception as e :
        NetworkSecurityException(e , sys) 
