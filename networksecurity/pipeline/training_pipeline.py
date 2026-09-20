import os , sys

from networksecurity.cloud.s3_syncer import S3Sync
from networksecurity.constant.training_pipeline import TRAINING_BUCKET_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.component.data_ingestion import DataIngestion
from networksecurity.component.data_validation import DataValidation
from networksecurity.component.data_transformation import DataTransformation
from networksecurity.component.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig , 
    DataIngestionConfig ,
    DataTransformationConfig ,
    DataValidationConfig ,
    ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifacts , 
    DataTransformationArtifacts , 
    DataValidationArtifacts ,
    ModelTrainerArtifacts
)

class TrainingPipeline :
    def __init__(self) -> None:
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()

    def start_data_integration(self) :
        try :
            self.data_integration_config = DataIngestionConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info("Start data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_integration_config)

            data_ingestion_artifacts : DataIngestionArtifacts = data_ingestion.inititate_data_ingestion()

            logging.info(f"Data ingestion compeleted and artifacts {data_ingestion_artifacts}")

            return data_ingestion_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys) 

    def start_data_validation(self , data_ingestion_artifacts : DataIngestionArtifacts) :
        try:
            self.data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data validation")
            data_validation = DataValidation(data_ingestion_artifacts=data_ingestion_artifacts , data_validation_config=self.data_validation_config)

            data_validation_artifacts : DataValidationConfig = data_validation.initiate_data_validation() 

            logging.info(f"Data validation completed and artifacts {data_validation_artifacts}") 

            return data_validation_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys)
    
    def start_data_transformation(self , data_validation_artifacts : DataValidationArtifacts) :
        try :
            self.data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data transformer")
            data_transformer = DataTransformation(data_validation_artifacts=data_validation_artifacts , data_transformation_config=self.data_transformation_config)

            data_transformation_artifacts : DataTransformationArtifacts = data_transformer.initiate_data_transformation() 

            logging.info(f"Data transforamation completed and artifacts {data_transformation_artifacts}") 

            return data_transformation_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys) 

    def start_model_trainer(self , data_transformer_artifacts : DataTransformationArtifacts) :
        try :
            self.model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data transformer")
            model_trainer = ModelTrainer(data_transformation_artifacts=data_transformer_artifacts , model_trainer_config=self.model_trainer_config)

            model_trainer_artifacts : ModelTrainerArtifacts = model_trainer.initiate_model_trainer() 

            logging.info(f"Model trainer completed and artifacts {model_trainer_artifacts}") 

            return model_trainer_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys) 

    def sync_artifacts_dir_to_S3(self) :
        try :
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir , aws_bucket_url=aws_bucket_url)

        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def sync_saved_model_dir_to_S3(self) :
        try :
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir , aws_bucket_url=aws_bucket_url)

        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def run_pipeline(self) :
        try :
            data_ingestion_artifacts = self.start_data_integration()
            data_validation_artifacts = self.start_data_validation(data_ingestion_artifacts=data_ingestion_artifacts)
            data_transformation_artifacts = self.start_data_transformation(data_validation_artifacts=data_validation_artifacts)
            model_trainer_artifacts = self.start_model_trainer(data_transformer_artifacts=data_transformation_artifacts)

            self.sync_artifacts_dir_to_S3()
            self.sync_saved_model_dir_to_S3()
            
        except Exception as e :
            raise NetworkSecurityException(e , sys) 