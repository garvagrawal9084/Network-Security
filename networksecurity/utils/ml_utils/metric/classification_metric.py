from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score , precision_score , recall_score
from networksecurity.logging.logger import logging
import sys

def get_classification_score(y_true , y_pred) -> ClassificationMetricArtifact :
    try :
        logging.info("Inside the get_classification_score")

        model_f1_score = f1_score(y_true=y_true , y_pred=y_pred)
        model_precision_score = precision_score(y_true=y_true , y_pred=y_pred)
        model_recall_score = recall_score(y_true=y_true , y_pred=y_pred)


        classificaion_metric = ClassificationMetricArtifact(f1_score=model_f1_score , precision_score=model_precision_score , recall_call=model_recall_score)

        return classificaion_metric

    except Exception as e :
        raise NetworkSecurityException(e , sys) 