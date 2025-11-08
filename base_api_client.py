# base_api_client.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any


class BaseAPIClient(ABC):
    @abstractmethod
    def fetch_data(self, features):
        """Fetch data based on pre-parsed features."""
        pass

    @abstractmethod
    def parse_response(self, response_package):
        """Parse and standardize the API response.
           Returns a tuple: (results, gathered_features)
        """
        pass
    
    
    # @abstractmethod
    # def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
    #     """
    #     Compute statistics on the DataFrame.
    #     Returns a dictionary with computed metrics (e.g. descriptive stats, missing values, etc.)
    #     The implementation can be customized per API client.
    #     """
    #     pass
    