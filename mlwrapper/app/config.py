import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', '5000'))

    # External ML Service (provided by Data Science team)
    ML_SERVICE_URL = os.getenv(
        'ML_SERVICE_URL', 'http://ml-service:8000/predict')
    ML_SERVICE_TIMEOUT = int(os.getenv('ML_SERVICE_TIMEOUT', '30'))

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
