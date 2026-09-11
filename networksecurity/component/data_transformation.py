import sys

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
)

from networksecurity.entity.artifact_entity import (
    DataValidationArtifacts,
    DataTransformationArtifacts,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataTransformationConfig

from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifacts: DataValidationArtifacts,
        data_transformation_config: DataTransformationConfig,
    ):

        try:
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:

        try:
            return pd.read_csv(file_path)

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def get_data_transformer_object() -> Pipeline:

        logging.info(
            "Enter get_data_transformer_object method"
        )

        try:

            # Initialize KNN Imputer
            imputer = KNNImputer(
                **DATA_TRANSFORMATION_IMPUTER_PARAMS
            )

            logging.info(
                f"Initialized KNN Imputer with parameters: "
                f"{DATA_TRANSFORMATION_IMPUTER_PARAMS}"
            )

            # Create preprocessing pipeline
            processor = Pipeline(
                [
                    ("imputer", imputer)
                ]
            )

            logging.info(
                "KNN Imputer pipeline created successfully"
            )

            return processor

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(
        self,
    ) -> DataTransformationArtifacts:

        logging.info(
            "Entered initiate_data_transformation method "
            "of DataTransformation class"
        )

        try:

            logging.info("Starting data transformation")

            # Read train and test data
            logging.info(
                "Reading train and test dataframe "
                "from data validation"
            )

            train_df = DataTransformation.read_data(
                self.data_validation_artifacts.valid_train_file_path
            )

            test_df = DataTransformation.read_data(
                self.data_validation_artifacts.valid_test_file_path
            )

            logging.info(
                "Train and test dataframe reading completed"
            )

            # -------------------------------
            # Training Data
            # -------------------------------

            logging.info("Preparing training dataframe")

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_train_df = train_df[TARGET_COLUMN]

            # Convert target -1 to 0
            target_feature_train_df = target_feature_train_df.replace(
                -1, 0
            )

            logging.info(
                "Training dataframe preparation completed"
            )

            # -------------------------------
            # Testing Data
            # -------------------------------

            logging.info("Preparing testing dataframe")

            # Separate input features and target
            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN]
            )

            target_feature_test_df = test_df[TARGET_COLUMN]

            # Convert target -1 to 0
            target_feature_test_df = target_feature_test_df.replace(
                -1, 0
            )

            logging.info(
                "Testing dataframe preparation completed"
            )

            # -------------------------------
            # Data Transformation
            # -------------------------------

            logging.info(
                "Creating data transformation object"
            )

            preprocessor = (
                DataTransformation.get_data_transformer_object()
            )

            logging.info(
                "Fitting data transformation object"
            )

            preprocessor_obj = preprocessor.fit(
                input_feature_train_df
            )

            logging.info(
                "Data transformation object fitted successfully"
            )

            # Transform training features
            transformed_input_train_features = (
                preprocessor_obj.transform(
                    input_feature_train_df
                )
            )

            # Transform testing features
            transformed_input_test_features = (
                preprocessor_obj.transform(
                    input_feature_test_df
                )
            )

            logging.info(
                "Train and test features transformed successfully"
            )

            # -------------------------------
            # Combine Features and Target
            # -------------------------------

            train_arr = np.c_[
                transformed_input_train_features,
                np.array(target_feature_train_df),
            ]

            test_arr = np.c_[
                transformed_input_test_features,
                np.array(target_feature_test_df),
            ]

            logging.info(
                "Train and test arrays created successfully"
            )

            # -------------------------------
            # Save Transformed Data
            # -------------------------------

            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr,
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr,
            )

            # Save preprocessing object
            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_obj,
            )

            logging.info(
                "Transformed data and preprocessing object saved successfully"
            )

            # -------------------------------
            # Create Artifacts
            # -------------------------------

            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_object_file_path=(
                    self.data_transformation_config
                    .transformed_object_file_path
                ),

                transformed_train_file_path=(
                    self.data_transformation_config
                    .transformed_train_file_path
                ),

                transformed_test_file_path=(
                    self.data_transformation_config
                    .transformed_test_file_path
                ),
            )

            logging.info(
                "Data transformation completed successfully"
            )

            return data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)