from narwhals import Object
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import RandomizedSearchCV
import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os , sys
import numpy as np
import dill
import pickle

def read_yaml_file(file_path : str) -> dict :
    try :
        with open(file_path , "rb") as file :
            return yaml.safe_load(file)
    except Exception as e :
        raise NetworkSecurityException(e , sys) 

def write_yaml_file(file_path : str , content : object , replace : bool = False) -> None:
    try :
        if replace :
            if os.path.exists(file_path) :
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path , "w") as file :
            yaml.dump(content, file)
    except Exception as e :
        raise NetworkSecurityException(e , sys) 


def save_numpy_array_data(file_path: str, array: np.ndarray):
    try:
        dir_name = os.path.dirname(file_path)

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        np.save(file_path, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_object(file_path : str , obj : Object) :
    try :
        logging.info("Entered the save_object method in main_utils")
        os.makedirs(os.path.dirname(file_path) , exist_ok=True)
        with open(file_path , "wb") as file :
            pickle.dump(obj, file)
        logging.info("Exited the save_object from main_utils")
    except Exception as e :
        raise NetworkSecurityException(e , sys)

def load_object(file_path : str) -> object:
    try :
        if not os.path.exists(file_path) :
            raise Exception(f"The file : {file_path} do not exist")
        with open(file_path , "rb") as file :
            print(file)
            return pickle.load(file)
    except Exception as e :
        raise NetworkSecurityException(e , sys)

def load_numpy_array_data(file_path : str) -> np.array :
    try :
        if not os.path.exists(file_path) :
            raise Exception(f"The file : {file_path} do not exist")
        
        return np.load(file_path)

    except Exception as e :
        raise NetworkSecurityException(e , sys)

def evaluate_model(
    X_train,
    y_train,
    X_test,
    y_test,
    models: dict,
    params: dict
):

    logging.info("Inside the evaluation model")

    try:

        test_report = {}
        train_report = {}
        best_models = {}

        for model_name, model in models.items():

            logging.info(f"RS for {model_name}")

            para = params[model_name]

            rs = RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                cv=5,
                n_jobs=-1,
                random_state=42
            )

            rs.fit(X_train, y_train)

            logging.info(f"RS for {model_name} end")

            logging.info("Getting best estimator")

            best_model = rs.best_estimator_

            logging.info(
                f"Best estimator obtained for {model_name}: "
                f"{best_model}"
            )

            logging.info("Starting train prediction")

            y_train_pred = best_model.predict(X_train)

            logging.info("Train prediction completed")

            logging.info("Starting test prediction")

            y_test_pred = best_model.predict(X_test)

            logging.info("Test prediction completed")

            logging.info("Calculating train accuracy")

            train_model_score = accuracy_score(
                y_true=y_train,
                y_pred=y_train_pred
            )

            logging.info("Calculating test accuracy")

            test_model_score = accuracy_score(
                y_true=y_test,
                y_pred=y_test_pred
            )

            train_report[model_name] = train_model_score
            test_report[model_name] = test_model_score

            # Store fitted best estimator
            best_models[model_name] = best_model

            logging.info(
                f"{model_name} - "
                f"Train Accuracy: {train_model_score}, "
                f"Test Accuracy: {test_model_score}"
            )

        return test_report, train_report, best_models

    except Exception as e:

        raise NetworkSecurityException(e, sys)