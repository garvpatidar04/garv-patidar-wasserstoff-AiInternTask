from pymongo import MongoClient
import logging
import os
from dotenv import load_dotenv

load_dotenv('.env')

database_url = os.getenv('database_url')
database_name = os.getenv('database_name')
database_collection = os.getenv('database_collection')
# MongoDB Setup
try:
    logging.basicConfig(level=logging.INFO, filename="app_ske.log")
    logger = logging.getLogger(__name__)
    
    client = MongoClient(database_url)
    db = client[database_name]
    collection = db[database_collection]
    logging.info("Connected to MongoDB successfully")
except Exception as e:
    logging.error(f"Connection to database failed : {e}")