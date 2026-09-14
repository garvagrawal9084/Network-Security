import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    ClassificationMetricArtifact,
    DataTransformationArtifacts,
    ModelTrainerArtifacts
)

from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    evaluate_model,
    save_numpy_array_data,
    load_numpy_array_data
)

from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier

import mlflow


class ModelTrainer:

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifacts: DataTransformationArtifacts
    ) -> None:

        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ---------------------------------------------------------
    # MLflow Tracking
    # ---------------------------------------------------------

    def track_mlflow(
    self,
    best_model,
    classification_train_metric: ClassificationMetricArtifact,
    classification_test_metric: ClassificationMetricArtifact):

        try:

            logging.info("Setting MLflow tracking URI")

            mlflow.set_tracking_uri(
                "http://127.0.0.1:5000"
            )

            logging.info("Setting MLflow experiment")

            mlflow.set_experiment(
                "Network Security"
            )

            logging.info("Starting MLflow run")

            with mlflow.start_run():

                logging.info("MLflow run started")

                # Training metrics
                logging.info("Logging train F1")

                mlflow.log_metric(
                    "train_f1_score",
                    classification_train_metric.f1_score
                )

                logging.info("Logging train precision")

                mlflow.log_metric(
                    "train_precision",
                    classification_train_metric.precision_score
                )

                logging.info("Logging train recall")

                mlflow.log_metric(
                    "train_recall",
                    classification_train_metric.recall_score
                )

                # Test metrics
                logging.info("Logging test F1")

                mlflow.log_metric(
                    "test_f1_score",
                    classification_test_metric.f1_score
                )

                logging.info("Logging test precision")

                mlflow.log_metric(
                    "test_precision",
                    classification_test_metric.precision_score
                )

                logging.info("Logging test recall")

                mlflow.log_metric(
                    "test_recall",
                    classification_test_metric.recall_score
                )

                logging.info("All metrics logged")

                # Model
                logging.info("Starting model logging")

                try:
                    mlflow.sklearn.log_model(
                        best_model,
                        name="best_model"
                    )
                    logging.info("Model logging completed")

                except Exception as e:
                    logging.exception("Model logging failed")
                    raise

                logging.info("Model logging completed")

            logging.info("MLflow run completed")

        except Exception as e:

            raise NetworkSecurityException(e,sys)

    # ---------------------------------------------------------
    # Model Training
    # ---------------------------------------------------------

    def train_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test
    ):

        try:

            # -------------------------------------------------
            # Define Models
            # -------------------------------------------------

            models = {

                "Random Forest": RandomForestClassifier(),

                "Decision Tree": DecisionTreeClassifier(),

                "Gradient Boosting": GradientBoostingClassifier(),

                "Xgboost": XGBClassifier(),

                "Logistic Regression": LogisticRegression(),

                "AdaBoost": AdaBoostClassifier(),

                "KNN": KNeighborsClassifier()
            }

            # -------------------------------------------------
            # Define Hyperparameter Grids
            # -------------------------------------------------

            params = {

                "Random Forest": {

                    "n_estimators": [100, 200],

                    "max_depth": [
                        None,
                        10,
                        20
                    ],

                    "min_samples_split": [
                        2,
                        5
                    ],

                    "min_samples_leaf": [
                        1,
                        2
                    ]
                },

                "Decision Tree": {

                    "criterion": [
                        "gini",
                        "entropy"
                    ],

                    "max_depth": [
                        None,
                        5,
                        10,
                        20
                    ],

                    "min_samples_split": [
                        2,
                        5,
                        10
                    ],

                    "min_samples_leaf": [
                        1,
                        2,
                        4
                    ],

                    "max_features": [
                        "sqrt",
                        "log2"
                    ]
                },

                "Gradient Boosting": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "max_depth": [
                        3,
                        5,
                        7
                    ]
                },

                "Xgboost": {

                    "n_estimators": [
                        100,
                        200
                    ],

                    "learning_rate": [
                        0.01,
                        0.05,
                        0.1
                    ],

                    "max_depth": [
                        3,
                        5,
                        7
                    ],

                    "subsample": [
                        0.8,
                        1.0
                    ],

                    "colsample_bytree": [
                        0.8,
                        1.0
                    ]
                },

                "Logistic Regression": {

                    "C": [
                        0.01,
                        0.1,
                        1,
                        10
                    ],

                    "solver": [
                        "lbfgs",
                        "liblinear"
                    ],

                    "max_iter": [
                        100,
                        200,
                        500
                    ]
                },

                "AdaBoost": {

                    "n_estimators": [
                        50,
                        100,
                        200
                    ],

                    "learning_rate": [
                        0.01,
                        0.1,
                        1.0
                    ]
                },

                "KNN": {

                    "n_neighbors": [
                        3,
                        5,
                        7,
                        9
                    ],

                    "weights": [
                        "uniform",
                        "distance"
                    ],

                    "metric": [
                        "euclidean",
                        "manhattan"
                    ]
                }
            }

            # -------------------------------------------------
            # Evaluate Models
            # -------------------------------------------------

            test_report: dict
            train_report: dict

            test_report, train_report, best_models = evaluate_model(

                X_train=X_train,

                y_train=y_train,

                X_test=X_test,

                y_test=y_test,

                models=models,

                params=params
            )

            logging.info(
                "Out of evaluate_model"
            )

            logging.info(
                "Started model evaluation and model selection"
            )

            # -------------------------------------------------
            # Sort Test Report
            # -------------------------------------------------

            test_report = dict(
                sorted(
                    test_report.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

            logging.info(
                f"Test report after sorting: {test_report}"
            )

            # -------------------------------------------------
            # Select Best Model
            # -------------------------------------------------

            best_model_name, best_model_score = list(
                test_report.items()
            )[0]

            logging.info(
                f"Best model found: "
                f"{best_model_name} with score: "
                f"{best_model_score}"
            )

            best_model = best_models[
                best_model_name
            ]

            logging.info(
                f"Best model object obtained: {best_model}"
            )

            logging.info(
                f"Best model parameters: "
                f"{best_model.get_params()}"
            )

            # -------------------------------------------------
            # Train Prediction
            # -------------------------------------------------

            logging.info(
                "Generating predictions on training dataset"
            )

            logging.info(
                f"X_train type: {type(X_train)}"
            )

            logging.info(
                f"X_train shape: {X_train.shape}"
            )

            logging.info(
                f"X_train memory: "
                f"{X_train.nbytes / (1024 ** 2):.2f} MB"
            )

            logging.info(
                "Starting training prediction"
            )

            y_train_pred = best_model.predict(
                X_train
            )

            logging.info(
                f"Training predictions completed. "
                f"Prediction shape: {y_train_pred.shape}"
            )

            # -------------------------------------------------
            # Training Metrics
            # -------------------------------------------------

            classification_train_metric: ClassificationMetricArtifact = (
                get_classification_score(
                    y_true=y_train,
                    y_pred=y_train_pred
                )
            )

            logging.info(
                "get_classification_score completed for training data"
            )

            logging.info(
                f"Training metric: "
                f"{classification_train_metric}"
            )

            # -------------------------------------------------
            # Test Prediction
            # -------------------------------------------------

            logging.info(
                "Generating predictions on test dataset"
            )

            y_test_pred = best_model.predict(
                X_test
            )

            logging.info(
                "Test predictions completed"
            )

            # -------------------------------------------------
            # Test Metrics
            # -------------------------------------------------

            classification_test_metric: ClassificationMetricArtifact = (
                get_classification_score(
                    y_true=y_test,
                    y_pred=y_test_pred
                )
            )

            logging.info(
                "get_classification_score completed for test data"
            )

            logging.info(
                f"Test metric: "
                f"{classification_test_metric}"
            )

            # -------------------------------------------------
            # Track Experiment with MLflow
            # -------------------------------------------------

            logging.info(
                "Starting MLflow experiment tracking"
            )

            self.track_mlflow(
                best_model,
                classification_train_metric,
                classification_test_metric
            )

            logging.info(
                "MLflow experiment tracking completed"
            )

            # -------------------------------------------------
            # Load Preprocessor
            # -------------------------------------------------

            logging.info(
                "Loading preprocessing object"
            )

            preprocessor = load_object(
                file_path=(
                    self.data_transformation_artifacts
                    .transformed_object_file_path
                )
            )

            logging.info(
                "Preprocessing object loaded successfully"
            )

            # -------------------------------------------------
            # Create Model Directory
            # -------------------------------------------------

            model_dir_path = os.path.dirname(
                self.model_trainer_config
                .trained_model_file_path
            )

            logging.info(
                f"Creating model directory at: "
                f"{model_dir_path}"
            )

            os.makedirs(
                model_dir_path,
                exist_ok=True
            )

            logging.info(
                "Model directory created successfully"
            )

            # -------------------------------------------------
            # Create NetworkModel
            # -------------------------------------------------

            logging.info(
                "Creating NetworkModel with "
                "preprocessor and best model"
            )

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=best_model
            )

            logging.info(
                "NetworkModel created successfully"
            )

            # -------------------------------------------------
            # Save Model
            # -------------------------------------------------

            logging.info(
                f"Saving trained model at: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                obj=network_model
            )

            logging.info(
                "Trained model saved successfully"
            )

            # -------------------------------------------------
            # Model Trainer Artifacts
            # -------------------------------------------------

            model_trainer_artifacts = ModelTrainerArtifacts(

                trained_model_file_path=(
                    self.model_trainer_config
                    .trained_model_file_path
                ),

                train_metric_artifacts=(
                    classification_train_metric
                ),

                test_metric_artifacts=(
                    classification_test_metric
                )
            )

            logging.info(
                f"Model trainer artifacts created: "
                f"{model_trainer_artifacts}"
            )

            logging.info(
                "Model training process completed successfully"
            )

            return model_trainer_artifacts

        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            )

    # ---------------------------------------------------------
    # Initiate Model Trainer
    # ---------------------------------------------------------

    def initiate_model_trainer(
        self
    ) -> ModelTrainerArtifacts:

        try:

            logging.info(
                "Initiate model trainer"
            )

            # -------------------------------------------------
            # Read Train and Test Data
            # -------------------------------------------------

            logging.info(
                "Read train and test data"
            )

            train_file_path = (
                self.data_transformation_artifacts
                .transformed_train_file_path
            )

            test_file_path = (
                self.data_transformation_artifacts
                .transformed_test_file_path
            )

            train_arr = load_numpy_array_data(
                train_file_path
            )

            test_arr = load_numpy_array_data(
                test_file_path
            )

            logging.info(
                "Read from train and test done"
            )

            # -------------------------------------------------
            # Split Features and Target
            # -------------------------------------------------

            X_train, y_train, X_test, y_test = (

                train_arr[:, :-1],

                train_arr[:, -1],

                test_arr[:, :-1],

                test_arr[:, -1]
            )

            # -------------------------------------------------
            # Train Model
            # -------------------------------------------------

            model_trainer_artifacts = self.train_model(
                X_train,
                y_train,
                X_test,
                y_test
            )

            return model_trainer_artifacts

        except Exception as e:

            raise NetworkSecurityException(
                e,
                sys
            )