from pymongo import MongoClient
from ..config import settings

# Initialize MongoDB client
mongo_client = MongoClient(settings.mongo_uri)[settings.mongo_db]

__all__ = ['mongo_client']
