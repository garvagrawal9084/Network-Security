from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# configuration of the data ingestion config

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifacts

import os
import sys
import numpy as np
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URI = os.getenv("MONGODB_URI")

class DataIngestion:
    def __init__(self ,  data_ingestion_config : DataIngestionConfig) -> None:
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e , sys)

    def export_collection_as_dataframe(self) :

        """
        Read data from mongodb
        """
        

        try:

            logging.info("Start the export_collection_as_dataframe")

            database_name = self.data_ingestion_config.database_name 
            collection_name = self.data_ingestion_config.collection_name

            logging.info("Got the db and collection name") 

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URI)

            logging.info("mongo connection done") 

            collection = self.mongo_client[database_name][collection_name]

            logging.info("mongo collection done") 


            logging.info("Making dataframe from mongodb") 

            df = pd.DataFrame(list(collection.find()))

            logging.info("completed dataframe from mongodb") 
            

            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
                logging.info("Removed _id column from dataframe")

            df.replace({"na" : np.nan} , inplace=True)

            logging.info("Remove nan")
            
            return df

        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def export_data_into_feature_store(self , dataframe : pd.DataFrame) :
        try:
            feature_path = self.data_ingestion_config.feature_store_file_path ; 
            dir_path = os.path.dirname(feature_path)
            os.makedirs(dir_path , exist_ok=True)
            dataframe.to_csv(feature_path , index=False , header=True)
            return dataframe
        except Exception as e :
            raise NetworkSecurityException(e , sys)

    def split_data_as_train_test(self , dataframe : pd.DataFrame) :
        try :
            train_set  , test_set = train_test_split(dataframe , test_size=self.data_ingestion_config.train_test_split_raio)

            logging.info("Performed train test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.data_ingestion_dir) 

            os.makedirs(dir_path , exist_ok=True)

            logging.info("Exporting train and test set into ingestion dir")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path , exist_ok=True)

            logging.info(f"Training file path: {self.data_ingestion_config.training_file_path}")
            train_set.to_csv(self.data_ingestion_config.training_file_path , index=False , header = True)

            test_set.to_csv(self.data_ingestion_config.test_file_path , index = False , header =True)

            logging.info("Export train and test file completed")

        except Exception as e :
            raise NetworkSecurityException(e , sys)             
    

    
    def inititate_data_ingestion(self) :
        try:
            dataframe = self.export_collection_as_dataframe() 
            dataframe = self.export_data_into_feature_store(dataframe= dataframe) 
            self.split_data_as_train_test(dataframe)

            data_ingestion_artifacts = DataIngestionArtifacts(trained_file_path=self.data_ingestion_config.training_file_path
            , test_file_path=self.data_ingestion_config.test_file_path)

            return data_ingestion_artifacts

        except Exception as e :
            raise NetworkSecurityException(e , sys)

    