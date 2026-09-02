from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import os
import sys
import certifi
import json
import pandas as pd
import numpy as np
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException

load_dotenv()


print(certifi.where())

class NetworkDataExtracy() :
    def __init__(self) -> None:
        try :
            pass
        except Exception as e :
            raise NetworkSecurityException(e,  sys)

    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)

            data.reset_index(drop=True, inplace=True)

            records = data.to_dict(orient="records")

            return records

        except Exception as e:
            raise NetworkSecurityException(e, sys)
    

    def insert_data_mongodb(self, records , database , collection) :
        try :
            self.database = database 
            self.collection = collection
            self.records = records

            self.mongo_client = MongoClient(os.getenv("MONGODB_URI"))
            self.database = self.mongo_client[self.database]

            self.collection = self.database[self.collection]

            self.collection.insert_many(self.records)

            return len(self.records)
        
        except Exception as e :
            raise NetworkSecurityException(e , sys)


if __name__ == "__main__" :
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "networkSecuity" 
    COLLECTION = "NetworkData"

    network_obj  = NetworkDataExtracy()
    records = network_obj.csv_to_json_convertor(file_path=FILE_PATH)
    no_of_records = network_obj.insert_data_mongodb(records , DATABASE , COLLECTION)

    print(no_of_records)