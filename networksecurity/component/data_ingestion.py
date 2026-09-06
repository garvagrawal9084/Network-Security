from nt import execv
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# configuration of the data ingestion config

from networksecurity.entity.config_entity import DataIngestionConfig

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
            database_name = self.data_ingestion_config.database_name 
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URI)

            collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list() :
                df.drop(columns=["_id"] , axis=1 , inplace=True)
            
            df.replace({"na" : np.nan} , inplace=True)

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
        except Exception as e :
            raise NetworkSecurityException(e , sys)

    