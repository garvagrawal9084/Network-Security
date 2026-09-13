import os , sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import ClassificationMetricArtifact, DataTransformationArtifacts , ModelTrainerArtifacts
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import save_object , load_object , evaluate_model
from networksecurity.utils.main_utils.utils import save_numpy_array_data , load_numpy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import   LogisticRegression 
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV



class ModelTrainer :
    def __init__(self , model_trainer_config : ModelTrainerConfig , data_transformation_artifacts : DataTransformationArtifacts) -> None:
        try :
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts
        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def train_model(self , X_train , y_train , X_test , y_test) :
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Xgboost": XGBClassifier(),
            "Logistic Regression": LogisticRegression(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "KNN": KNeighborsClassifier()
        }

        params = {
            "Random Forest": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },

            "Decision Tree": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features" : ["sqrt" , "log2"]
            },

            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            },

            "Xgboost": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            },

            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [100, 200, 500]
            },

            "AdaBoost": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 1.0]
            },

            "KNN": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        }

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

        logging.info("Out of evaluate_model")

        logging.info("Started model evaluation and model selection")

        test_report = dict(
            sorted(
                test_report.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )

        logging.info(f"Test report after sorting: {test_report}")

        best_model_name, best_model_score = list(test_report.items())[0]

        logging.info(
            f"Best model found: "
            f"{best_model_name} with score: "
            f"{best_model_score}"
        )

        best_model = best_models[best_model_name]

        best_model.set_params(n_jobs=-1 , verbose = 0)

        logging.info(
            f"Best model object obtained: {best_model}"
        )

        # Train prediction
        logging.info("Generating predictions on training dataset")

        logging.info(
            f"X_train type: {type(X_train)}"
        )

        logging.info(
            f"Best model parameters: {best_model.get_params()}"
        )

        logging.info("Starting training prediction")
        logging.info(f"X_train shape: {X_train.shape}")
        logging.info(f"X_train memory: {X_train.nbytes / (1024 ** 2):.2f} MB")


        y_train_pred = best_model.predict(X_train)

        logging.info(
            f"Training predictions completed. "
            f"Prediction shape: {y_train_pred.shape}"
        )

        logging.info("Training predictions completed")

        classification_train_metric: ClassificationMetricArtifact = (
            get_classification_score(
                y_true=y_train,
                y_pred=y_train_pred
            )
        )

        logging.info(
            f"Training metric: {classification_train_metric}"
        )

        # Test prediction
        logging.info("Generating predictions on test dataset")

        y_test_pred = best_model.predict(X_test)

        logging.info("Test predictions completed")

        classification_test_metric: ClassificationMetricArtifact = (
            get_classification_score(
                y_true=y_test,
                y_pred=y_test_pred
            )
        )

        logging.info(
            f"Test metric: {classification_test_metric}"
        )

        # Load preprocessor
        logging.info("Loading preprocessing object")

        preprocessor = load_object(
            file_path=self.data_transformation_artifacts.transformed_object_file_path
        )

        logging.info("Preprocessing object loaded successfully")

        # Create model directory
        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )

        logging.info(
            f"Creating model directory at: {model_dir_path}"
        )

        os.makedirs(
            model_dir_path,
            exist_ok=True
        )

        logging.info("Model directory created successfully")

        # Create NetworkModel
        logging.info(
            "Creating NetworkModel with preprocessor and best model"
        )

        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model
        )

        logging.info("NetworkModel created successfully")

        # Save model
        logging.info(
            f"Saving trained model at: "
            f"{self.model_trainer_config.trained_model_file_path}"
        )

        save_object(
            self.model_trainer_config.trained_model_file_path,
            obj=network_model
        )

        logging.info("Trained model saved successfully")

        # Model trainer artifacts
        model_trainer_artifacts = ModelTrainerArtifacts(
            trained_model_file_path=(
                self.model_trainer_config.trained_model_file_path
            ),
            train_metric_artifacts=classification_train_metric,
            test_metric_artifacts=classification_test_metric
        )

        logging.info(
            f"Model trainer artifacts created: {model_trainer_artifacts}"
        )

        logging.info("Model training process completed successfully")

        return model_trainer_artifacts


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

            model_trainer_artifacts = self.train_model(X_train , y_train , X_test , y_test)

            return model_trainer_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys)