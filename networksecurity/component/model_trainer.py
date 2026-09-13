import os , sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import DataTransformationArtifacts , ModelTrainerArtifacts
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object , load_object
from networksecurity.utils.main_utils.utils import save_numpy_array_data , load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

class ModelTrainer :
    def __init__(self , model_trainer_config : ModelTrainerConfig , data_transformation_artifacts : DataTransformationArtifacts) -> None:
        try :
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts
        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def train_model(self , X_train , y_train) :
        pass

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try :
            logging.info("Initiate model trainer")
            logging.info("Read train and test data")

            train_file_path = self.data_transformation_artifacts.transformed_train_file_path
            test_file_path = self.data_transformation_artifacts.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info("Read from train and test done")

            X_train , y_train , X_test , y_test = (
                train_arr[: , :-1],
                train_arr[: , -1],
                test_arr[: , :-1],
                test_arr[: , -1],
            )

            model = self.tra

        except Exception as e :
            raise NetworkSecurityException(e , sys)