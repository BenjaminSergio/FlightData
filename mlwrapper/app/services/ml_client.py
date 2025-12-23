import requests
from typing import Dict, Any
from app.config import Config
import logging

logger = logging.getLogger(__name__)


class MLServiceClient:
    """
    HTTP client for communication with external ML service

    This ML service is developed and maintained by the Data Science team.
    Our Flask API only acts as an intermediary between the Java API and the ML service.
    """

    def __init__(self):
        self.ml_service_url = Config.ML_SERVICE_URL
        self.timeout = Config.ML_SERVICE_TIMEOUT
        logger.info(f"MLServiceClient configured for: {self.ml_service_url}")

    def predict(self, flight_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends request to external ML service

        Args:
            flight_data: Flight data coming from Java API
            {
                "flightNumber": "AA1234",
                "companyName": "AA",
                "flightOrigin": "JFK",
                "flightDestination": "LAX",
                "flightDepartureDate": "2025-12-20T14:30:00",
                "flightDistance": 3974
            }

        Returns:
            ML service response:
            {
                "prediction": 0 or 1,  # 0 = ON_TIME, 1 = DELAYED
                "probability": 0.85     # Confidence (0.0 - 1.0)
            }

        Raises:
            Exception: If there's an error communicating with ML service
        """

        try:
            logger.info(
                f"Sending request to ML service: {flight_data.get('flightNumber')}"
            )

            # Make HTTP POST request to ML service
            response = requests.post(
                self.ml_service_url,
                json=flight_data,
                headers={'Content-Type': 'application/json'},
                timeout=self.timeout
            )

            # Check if request was successful
            response.raise_for_status()

            # Parse JSON response
            result = response.json()

            logger.info(
                f"Prediction received from ML service: "
                f"prediction={result.get('prediction')}, "
                f"probability={result.get('probability')}"
            )

            return result

        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to ML service")
            raise Exception("ML service did not respond in time")

        except requests.exceptions.ConnectionError:
            logger.error("Connection error with ML service")
            raise Exception("Could not connect to ML service")

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error from ML service: {e.response.status_code}")
            error_detail = e.response.json() if e.response.content else {}
            raise Exception(f"Error in ML service: {error_detail}")

        except Exception as e:
            logger.error(f"Unexpected error calling ML service: {str(e)}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Checks if external ML service is available
        """

        try:
            # Try to make request to health endpoint (adjust according to ML service API)
            health_url = self.ml_service_url.replace('/predict', '/health')
            response = requests.get(health_url, timeout=5)
            response.raise_for_status()
            return {"status": "UP", "ml_service": "OK"}
        except Exception as e:
            logger.warning(f"ML service health check failed: {e}")
            return {"status": "DOWN", "ml_service": str(e)}


# Singleton
_ml_client = None


def get_ml_client() -> MLServiceClient:
    """Returns singleton instance of ML client"""
    global _ml_client
    if _ml_client is None:
        _ml_client = MLServiceClient()
    return _ml_client
