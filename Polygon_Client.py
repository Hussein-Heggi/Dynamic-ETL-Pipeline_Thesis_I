# polygon_client.py
import time
import pandas as pd
from typing import Dict, Any, Tuple
from base_api_client import BaseAPIClient
from polygon import RESTClient

class PolygonClient(BaseAPIClient):
    def __init__(self, api_key: str):
        """
        Initializes the Polygon REST client and sets up endpoint descriptions.
        """
        self.client = RESTClient(api_key)
        self.endpoint_descriptions = {
            0: "Aggregated bars for a given ticker over a specified interval",
            1: "Grouped daily aggregates for the specified date",
            2: "Daily open/close aggregate for a given ticker and date",
            3: "Previous close aggregate for the given ticker"
        }

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data using the official client's methods.
        """
        endpoint_mapping = {
            0: lambda f: self.client.get_aggs(
                ticker=f['ticker'],
                multiplier=f['multiplier'],
                timespan=f['timespan'],
                from_=f['from'],
                to=f['to']
            ),
            1: lambda f: self.client.get_grouped_daily_aggs(
                date=f['from']
            ),
            2: lambda f: self.client.get_daily_open_close_agg(
                ticker=f['ticker'],
                date=f['from']
            ),
            3: lambda f: self.client.get_previous_close_agg(
                ticker=f['ticker']  
            )
        }
        endpoint_type = features.get('endpoint_type', 0)
        if endpoint_type not in endpoint_mapping:
            raise ValueError(f"Invalid endpoint_type: {endpoint_type}")

        attempts = 0
        max_attempts = 3
        delay = 2
        while attempts < max_attempts:
            try:
                response = endpoint_mapping[endpoint_type](features)
                return {'data': response, 'features': features}
            except Exception as e:
                attempts += 1
                if attempts == max_attempts:
                    print(f"Request failed after {max_attempts} attempts: {e}")
                    raise
                print(f"Attempt {attempts} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def parse_response(self, response_package: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Parses raw response into DataFrame and metadata dict.
        """
        raw = response_package['data']
        params = response_package['features']
        # Normalize records
        if not isinstance(raw, (dict, list)):
            raw = raw.__dict__
        records = raw.get('results', [raw]) if isinstance(raw, dict) else raw

        df = pd.DataFrame(records)


        return df

    # def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
    #     """
    #     Compute descriptive statistics on the DataFrame.
    #     """
    #     stats: Dict[str, Any] = {
    #         'descriptive_stats': df.describe().to_dict(),
    #         'missing_values': df.isnull().sum().to_dict(),
    #         'skewness': df.skew().to_dict(),
    #         'kurtosis': df.kurtosis().to_dict()
    #     }
    #     return stats
 