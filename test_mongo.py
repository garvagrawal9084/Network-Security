from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import os

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URI")) 

try:
    client.admin.command("ping")
    print("Pinged")
except Exception as e :
    print(e)
