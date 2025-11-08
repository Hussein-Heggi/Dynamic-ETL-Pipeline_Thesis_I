import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
from base_api_client import BaseAPIClient

class AlphaVantageClient(BaseAPIClient):
    """
    API Client for fetching stock data from Alpha Vantage.
    Inherits from BaseAPIClient and implements methods for fetching,
    parsing, and computing statistics for stock price data.
    """
    def __init__(self, api_key: str):
        """
        Initializes the AlphaVantageClient.

        Args:
            api_key (str): The API key for accessing Alpha Vantage services.
        """
        if not api_key:
            raise ValueError("API key is required for AlphaVantageClient")
        self.api_key = api_key
        self.base_url = 'https://www.alphavantage.co/query'
        # Define standard OHLCV columns expected/handled by this client
        self._standard_columns = ['open', 'high', 'low', 'close', 'volume']

    def fetch_data(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch stock price data from Alpha Vantage based on provided features.

        Args:
            features (Dict[str, Any]): A dictionary containing parameters for the API call.
                Expected keys:
                'ticker' (str): The stock symbol (e.g., 'IBM').
                'timespan' (str): The time interval ('1min', '5min', '15min', '30min', '60min', 'day', 'week', 'month').
                'outputsize' (str, optional): 'compact' or 'full'. Defaults vary by timespan.
                'month' (str, optional): Specific month for intraday data (YYYY-MM).

        Returns:
            Dict[str, Any]: A dictionary containing the raw JSON response data under the key 'data'
                            and the features dictionary used for the request under 'features'.

        Raises:
            ValueError: If the timespan is unsupported or if the API returns an error message.
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        params = {
            "symbol": features['ticker'],
            "apikey": self.api_key,
            "datatype": "json" # Request JSON format
        }

        timespan = features.get('timespan', 'day').lower() # Default to daily if not specified

        # Map user-friendly timespan to Alpha Vantage function names [1]
        if timespan == 'day':
            params['function'] = 'TIME_SERIES_DAILY'
            params['outputsize'] = features.get('outputsize', 'full') # Default to full history
        elif timespan == 'week':
            params['function'] = 'TIME_SERIES_WEEKLY'
            params['outputsize'] = features.get('outputsize', 'full') # Default to full history
        elif timespan == 'month':
            params['function'] = 'TIME_SERIES_MONTHLY'
            params['outputsize'] = features.get('outputsize', 'full') # Default to full history
        elif timespan in ['1min', '5min', '15min', '30min', '60min']:
            params['function'] = 'TIME_SERIES_INTRADAY'
            params['interval'] = timespan
            params['outputsize'] = features.get('outputsize', 'compact') # Default to recent 100 points
            # Add month parameter if provided for intraday history [1]
            if 'month' in features:
                params['month'] = features['month']
                params['outputsize'] = features.get('outputsize', 'full') # Need 'full' when specifying month
        else:
            raise ValueError(f"Unsupported timespan for Alpha Vantage: {timespan}")

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
            data = response.json()

            # Check for API-specific errors returned in the JSON [1, 5]
            if not data:
                 raise ValueError("Alpha Vantage API returned an empty response.")
            if "Error Message" in data:
                 raise ValueError(f"Alpha Vantage API Error: {data['Error Message']}")
            # Check for rate limit notes [1]
            if "Note" in data and "API call frequency" in data["Note"]:
                 print(f"Warning: Alpha Vantage rate limit may have been hit. Message: {data['Note']}")


            return {"data": data, "features": features}

        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed: {e}")
            raise
        except ValueError as e: # Catch explicit errors raised above
             print(f"Data fetch/validation error: {e}")
             raise
        except Exception as e: # Catch other potential errors (e.g., json decoding)
             print(f"An unexpected error occurred during data fetching: {e}")
             raise


    def parse_response(self, response_package: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Parse the raw JSON response from Alpha Vantage into a standardized Pandas DataFrame.

        Args:
            response_package (Dict[str, Any]): The dictionary returned by fetch_data, containing
                                               'data' (raw JSON) and 'features'.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing:
                - A Pandas DataFrame with datetime index and columns for the requested features (e.g., open, high, low, close, volume).
                - A list of strings representing the column names present in the returned DataFrame.

        Raises:
            ValueError: If the time series data cannot be found or parsed correctly.
        """
        data = response_package["data"]
        requested_features = response_package["features"]
        columns_to_include = requested_features.get('columns', self._standard_columns) # User specified columns or default OHLCV

        # Find the main time series data key (dynamic based on the function used) [5]
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break

        if not time_series_key:
             # Handle cases where API call succeeded but returned no time series (e.g., bad ticker symbol results in empty dict sometimes)
             if data.get("Meta Data"): # Check if metadata exists, implying a potentially valid but data-less response
                 print(f"Warning: No time series data found for ticker '{requested_features.get('ticker', 'N/A')}' with features {requested_features}. Returning empty DataFrame.")
                 return pd.DataFrame(), []
             else:
                 raise ValueError(f"Could not find time series data key in Alpha Vantage response: {list(data.keys())}")

        time_series_data = data[time_series_key]

        try:
            # Convert the dictionary of time points into a DataFrame
            df = pd.DataFrame.from_dict(time_series_data, orient='index')

            # Rename columns to remove Alpha Vantage prefixes (e.g., "1. open" -> "open") [5]
            df.rename(columns=lambda x: x.split('. ')[1] if '. ' in x else x, inplace=True)

            # Convert index to datetime objects and sort chronologically
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Select only the standard columns that are actually present after renaming
            available_standard_cols = [col for col in self._standard_columns if col in df.columns]
            df = df[available_standard_cols]

            # Convert OHLCV columns to numeric types, coercing errors to NaN
            for col in available_standard_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filter to only the columns requested by the user (if specified) that are available
            final_columns = [col for col in columns_to_include if col in df.columns]
            df = df[final_columns] # Keep only the requested & available columns

            return df

        except Exception as e:
            print(f"Error parsing Alpha Vantage response: {e}")
            raise ValueError(f"Failed to parse time series data. Error: {e}")


    # def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
    #     """
    #     Compute descriptive statistics on the parsed DataFrame.

    #     Args:
    #         df (pd.DataFrame): The DataFrame returned by parse_response.

    #     Returns:
    #         Dict[str, Any]: A dictionary containing computed statistics:
    #             - 'descriptive_stats': Basic statistics (mean, std, min, max, etc.).
    #             - 'missing_values': Count of NaN values per column.
    #             - 'skewness': Skewness of numeric columns.
    #             - 'kurtosis': Kurtosis of numeric columns.
    #             - 'shape': Tuple representing (rows, columns) of the DataFrame.
    #             - 'column_types': Data types of each column.
    #     """
    #     stats = {}
    #     numeric_df = df.select_dtypes(include='number') # Operate only on numeric columns

    #     if not numeric_df.empty:
    #         stats["descriptive_stats"] = numeric_df.describe().to_dict()
    #         stats["skewness"] = numeric_df.skew().to_dict()
    #         stats["kurtosis"] = numeric_df.kurtosis().to_dict()
    #     else:
    #         stats["descriptive_stats"] = {}
    #         stats["skewness"] = {}
    #         stats["kurtosis"] = {}

    #     stats["missing_values"] = df.isnull().sum().to_dict()
    #     stats["shape"] = df.shape
    #     stats["column_types"] = df.dtypes.astype(str).to_dict()

    #     return stats

# Example usage (ensure base_api_client.py is accessible)
if __name__ == "__main__":
    # NOTE: Replace 'demo' with your actual Alpha Vantage API key for extensive testing
    # The 'demo' key is heavily rate-limited and may not work reliably. [1]
    config = {'alpha_vantage_api_key': 'WXOG38FYIAUD05SZ'}

    client = AlphaVantageClient(api_key=config.get('alpha_vantage_api_key'))

    # --- Test Case 1: Daily Data (Default Columns) ---
    print("\n--- Testing Daily (Default Columns) ---")
    features_daily = {
        'ticker': 'IBM',
        'timespan': 'day',
        'outputsize': 'compact' # Use compact for demo key to reduce data size
    }
    try:
        raw_package_daily = client.fetch_data(features_daily)
        # print("Fetched Daily Raw Keys:", raw_package_daily['data'].keys()) # Inspect raw response structure
        df_daily, cols_daily = client.parse_response(raw_package_daily)
        print(f"Parsed Daily DataFrame Head ({df_daily.shape}):\n", df_daily.head())
        print("Daily Columns Returned:", cols_daily)
        stats_daily = client.compute_statistics(df_daily)
        print("Daily Stats (Shape):", stats_daily.get('shape'))
        print("Daily Stats (Missing Values):", stats_daily.get('missing_values'))
    except Exception as e:
        print(f"Daily Test Failed: {e}")

    # --- Test Case 2: Intraday Data (Specific Columns) ---
    print("\n--- Testing Intraday 5min (Specific Columns) ---")
    features_intraday = {
        'ticker': 'IBM',
        'timespan': '5min',
        'outputsize': 'compact',
        'columns': ['open', 'close', 'volume'] # Request specific columns
    }
    try:
        raw_package_intraday = client.fetch_data(features_intraday)
        df_intraday, cols_intraday = client.parse_response(raw_package_intraday)
        print(f"Parsed Intraday DataFrame Head ({df_intraday.shape}):\n", df_intraday.head())
        print("Intraday Columns Returned:", cols_intraday)
        stats_intraday = client.compute_statistics(df_intraday)
        print("Intraday Stats (Shape):", stats_intraday.get('shape'))
        print("Intraday Stats (Missing Values):", stats_intraday.get('missing_values'))
    except Exception as e:
        print(f"Intraday Test Failed: {e}")

    # --- Test Case 3: Monthly Data (History) ---
    print("\n--- Testing Monthly (Full History - may fail with demo key) ---")
    features_monthly = {
        'ticker': 'TSCO.LON', # Example for a non-US ticker [1]
        'timespan': 'month',
        'outputsize': 'full'
    }
    try:
        # This call might be large and could fail or be slow with a 'demo' key
        raw_package_monthly = client.fetch_data(features_monthly)
        df_monthly, cols_monthly = client.parse_response(raw_package_monthly)
        print(f"Parsed Monthly DataFrame Head ({df_monthly.shape}):\n", df_monthly.head())
        print("Monthly Columns Returned:", cols_monthly)
        stats_monthly = client.compute_statistics(df_monthly)
        print("Monthly Stats (Shape):", stats_monthly.get('shape'))
    except Exception as e:
        print(f"Monthly Test Failed: {e}")

    # --- Test Case 4: Invalid Ticker ---
    print("\n--- Testing Invalid Ticker ---")
    features_invalid = {
        'ticker': 'INVALIDTICKERXYZ',
        'timespan': 'day',
        'outputsize': 'compact'
    }
    try:
        raw_package_invalid = client.fetch_data(features_invalid)
        # Parsing might raise an error or return an empty DataFrame depending on API response
        df_invalid, cols_invalid = client.parse_response(raw_package_invalid)
        print(f"Parsed Invalid Ticker DataFrame ({df_invalid.shape}):\n", df_invalid.head())
        stats_invalid = client.compute_statistics(df_invalid)
        print("Invalid Ticker Stats (Shape):", stats_invalid.get('shape'))
    except Exception as e:
        print(f"Invalid Ticker Test completed (expected error or empty data): {e}")