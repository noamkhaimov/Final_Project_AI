from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any
from ..config import settings

_client = MongoClient(settings.mongo_uri)
db = _client[settings.mongo_db]

# Create collection if it doesn't exist
if "application_logs" not in db.list_collection_names():
    db.create_collection("application_logs")

coll = db["application_logs"]


class ApplicationLogger:
    """
    Custom logger class to handle application logging
    Logs to both file and database
    """

    def __init__(self, db_collection=coll):
        self.db_collection = db_collection

    def log_request(self, endpoint: str, method: str, status_code: int,
                   message: str, additional_data: Dict[str, Any] = None):
        """
        Log application request to database

        Args:
            endpoint: API endpoint that was called
            method: HTTP method used
            status_code: HTTP response status code
            message: Log message
            additional_data: Additional data to include in log
        """
        try:
            log_record = {
                'timestamp': datetime.utcnow(),
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'message': message,
                'additional_data': additional_data or {}
            }

            self.db_collection.insert_one(log_record)
        except Exception as e:
            # Don't let logging errors break the application
            print(f"Failed to log request to database: {e}")


# Create a global logger instance
app_logger = ApplicationLogger()
