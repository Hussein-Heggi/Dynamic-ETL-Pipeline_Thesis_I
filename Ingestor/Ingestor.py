# ingestor.py
import pandas as pd
import math
from typing import Dict, Any, List, Union
from Polygon_Client import PolygonClient
from alpha_vantage_client import AlphaVantageClient
from prompt_eng import endpointMatcher, api_endpoint_map
from datetime import datetime, timedelta

class Ingestor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Ingestor with a registry of API clients and endpoint matcher.
        Uses the api_endpoint_map imported from prompt_eng.
        """
        self.clients = {
            'polygon': PolygonClient(api_key=config.get('polygon_api_key')),
            'alpha_vantage': AlphaVantageClient(api_key=config.get('alpha_vantage_api_key')),
        }
        
        # Initialize the endpoint matcher with the mapping imported from prompt_eng
        self.endpoint_matcher = endpointMatcher(api_endpoint_map)
    
    def _map_to_client_format(self, api_name: str, endpoint_name: str, extracted_params: Dict) -> Dict[str, Any]:
        """
        Map the extracted parameters to the actual client parameter format.
        """
        mapped_params = extracted_params.copy()
        
        # Validate and fix date parameters
        today = datetime.now().date()
        
        # For polygon, add endpoint_type based on endpoint name
        if api_name == 'polygon':
            endpoint_type_map = {
                'get_aggs': 0,
                'get_grouped_daily_aggs': 1, 
                'get_daily_open_close_agg': 2,
                'get_previous_close_agg': 3
            }
            if endpoint_name in endpoint_type_map:
                mapped_params['endpoint_type'] = endpoint_type_map[endpoint_name]
            
            # Handle date parameters - ensure they're single strings, not lists
            if 'date' in mapped_params:
                date_value = mapped_params['date']
                if isinstance(date_value, list):
                    # Take the first date if multiple are provided
                    mapped_params['date'] = date_value[0]
                # Map 'date' to 'from' for polygon endpoints that use date
                if 'from' not in mapped_params:
                    mapped_params['from'] = mapped_params['date']
            
            # Validate dates are not in the future
            for date_param in ['date', 'from', 'to']:
                if date_param in mapped_params:
                    try:
                        param_date = datetime.strptime(mapped_params[date_param], '%Y-%m-%d').date()
                        if param_date > today:
                            # Replace future dates with today
                            mapped_params[date_param] = today.strftime('%Y-%m-%d')
                            print(f"Warning: {date_param} was in the future, changed to today")
                    except ValueError:
                        # If date parsing fails, use today
                        mapped_params[date_param] = today.strftime('%Y-%m-%d')
                        print(f"Warning: Invalid {date_param} format, using today")
            
            # Add required 'from' and 'to' parameters for get_aggs if missing
            if endpoint_name == 'get_aggs':
                if 'from' not in mapped_params:
                    # Default to last 30 days
                    mapped_params['from'] = (today - timedelta(days=30)).strftime('%Y-%m-%d')
                if 'to' not in mapped_params:
                    mapped_params['to'] = today.strftime('%Y-%m-%d')
        
        # Fix for Alpha Vantage parameters
        if api_name == 'alpha_vantage':
            # Ensure timespan is valid for intraday
            if endpoint_name == 'TIME_SERIES_INTRADAY' and 'timespan' in mapped_params:
                valid_intervals = ['1min', '5min', '15min', '30min', '60min']
                if mapped_params['timespan'] not in valid_intervals:
                    # Map common invalid values to valid ones
                    timespan_map = {
                        '1day': '60min', 'day': '60min', 'daily': '60min',
                        '1minute': '1min', '5minute': '5min', '15minute': '15min',
                        '30minute': '30min', '60minute': '60min'
                    }
                    mapped_params['timespan'] = timespan_map.get(mapped_params['timespan'], '5min')
        
        return mapped_params
    
    def process_natural_language(self, prompt: str) -> List[Dict[str, Any]]:
    
        print(f"Processing natural language prompt: {prompt}")
        
        # Use endpoint matcher to parse the prompt
        endpoint_matches, recommended_apis, api_endpoint_mapping, endpoint_params = \
            self.endpoint_matcher.match_prompt(prompt)
        
        # Select the best endpoint instead of using all matches
        feature_requests = self._select_best_feature_request(
            endpoint_matches, api_endpoint_mapping, endpoint_params, prompt
        )
        
        # Process the feature requests (should be only 1-2 at most now)
        return self.process_features(feature_requests)

    def _select_best_feature_request(self, endpoint_matches: Dict, api_endpoint_mapping: Dict, 
                                endpoint_params: Dict, prompt: str) -> List[Dict[str, Any]]:
        """
        Select the best endpoint from each API, considering frequency and scores.
        """
        if not endpoint_matches:
            print("No endpoint matches found for prompt")
            return []
        
        # Count endpoint frequencies and aggregate scores
        endpoint_scores = {}
        for keyword, match_info in endpoint_matches.items():
            endpoint_name = match_info['endpoint']
            score = match_info['score']
            apis = match_info['apis']
            
            for api_name in apis:
                key = (api_name, endpoint_name)
                if key not in endpoint_scores:
                    endpoint_scores[key] = {
                        'api': api_name,
                        'endpoint': endpoint_name,
                        'total_score': 0.0,
                        'count': 0,
                        'keywords': [],
                        'params': endpoint_params.get(endpoint_name, {}).get(api_name, {})
                    }
                endpoint_scores[key]['total_score'] += score
                endpoint_scores[key]['count'] += 1
                endpoint_scores[key]['keywords'].append(keyword)
        
        # Convert to list and calculate weighted scores
        scored_endpoints = []
        for key, endpoint_info in endpoint_scores.items():
            # Weighted score: average score * log(count) to favor frequently matched endpoints
            avg_score = endpoint_info['total_score'] / endpoint_info['count']
            frequency_bonus = math.log(endpoint_info['count'] + 1)  # +1 to avoid log(0)
            weighted_score = avg_score * frequency_bonus
            
            scored_endpoints.append({
                'api': endpoint_info['api'],
                'endpoint': endpoint_info['endpoint'],
                'weighted_score': weighted_score,
                'avg_score': avg_score,
                'count': endpoint_info['count'],
                'keywords': endpoint_info['keywords'],
                'params': endpoint_info['params']
            })
        
        # Sort by weighted score (highest first)
        scored_endpoints.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        print(f"Ranked endpoints (considering frequency and score):")
        for i, endpoint in enumerate(scored_endpoints[:8]):  # Show top 8
            print(f"  {i+1}. {endpoint['api']}.{endpoint['endpoint']} "
                f"(weighted: {endpoint['weighted_score']:.2f}, "
                f"count: {endpoint['count']}, keywords: {endpoint['keywords'][:3]})")
        
        # Select best endpoint from each API
        selected_requests = self._select_per_api(scored_endpoints, prompt)
        
        print(f"Selected {len(selected_requests)} endpoints from {len(set(r['api'] for r in selected_requests))} APIs")
        return selected_requests

    def _select_per_api(self, scored_endpoints: List[Dict], prompt: str) -> List[Dict[str, Any]]:
        """
        Select the best endpoint from each available API.
        """
        selected_requests = []
        used_apis = set()
        
        prompt_lower = prompt.lower()
        
        # Apply business rules to prioritize certain endpoints
        prioritized_endpoints = self._apply_business_rules(scored_endpoints, prompt_lower)
        
        # First, add any prioritized endpoints
        for endpoint in prioritized_endpoints:
            if endpoint['api'] not in used_apis:
                mapped_params = self._map_to_client_format(
                    endpoint['api'], endpoint['endpoint'], endpoint['params']
                )
                selected_requests.append({
                    'api': endpoint['api'],
                    **mapped_params
                })
                used_apis.add(endpoint['api'])
                print(f"Prioritized: {endpoint['api']}.{endpoint['endpoint']}")
        
        # Then, select the best from each remaining API
        for endpoint in scored_endpoints:
            if endpoint['api'] not in used_apis:
                mapped_params = self._map_to_client_format(
                    endpoint['api'], endpoint['endpoint'], endpoint['params']
                )
                selected_requests.append({
                    'api': endpoint['api'],
                    **mapped_params
                })
                used_apis.add(endpoint['api'])
                print(f"Selected: {endpoint['api']}.{endpoint['endpoint']}")
            
            # Limit to 2-3 APIs max to avoid too many requests
            if len(selected_requests) >= 3:
                break
        
        return selected_requests

    def _apply_business_rules(self, scored_endpoints: List[Dict], prompt_lower: str) -> List[Dict]:
        """
        Apply business rules to prioritize endpoints that match the prompt intent.
        """
        prioritized = []
        
        # Rule 1: Keyword-based prioritization
        keyword_priority = {
            'intraday': ['TIME_SERIES_INTRADAY', 'get_aggs'],
            'previous close': ['get_previous_close_agg'],
            'daily': ['TIME_SERIES_DAILY', 'get_daily_open_close_agg', 'get_aggs'],
            'weekly': ['TIME_SERIES_WEEKLY'],
            'monthly': ['TIME_SERIES_MONTHLY'],
            'grouped': ['get_grouped_daily_aggs'],
            'all stocks': ['get_grouped_daily_aggs'],
            'open close': ['get_daily_open_close_agg'],
            'aggregates': ['get_aggs', 'get_grouped_daily_aggs']
        }
        
        for keyword, preferred_endpoints in keyword_priority.items():
            if keyword in prompt_lower:
                # Find the highest-scoring endpoint that matches this keyword
                matching_endpoints = [
                    ep for ep in scored_endpoints 
                    if ep['endpoint'] in preferred_endpoints
                ]
                if matching_endpoints:
                    best_match = max(matching_endpoints, key=lambda x: x['weighted_score'])
                    if best_match not in prioritized:
                        prioritized.append(best_match)
        
        # Rule 2: Prefer endpoints that match multiple keywords from the prompt
        high_frequency_endpoints = [ep for ep in scored_endpoints if ep['count'] >= 2]
        if high_frequency_endpoints:
            best_high_freq = max(high_frequency_endpoints, key=lambda x: x['weighted_score'])
            if best_high_freq not in prioritized:
                prioritized.append(best_high_freq)
        
        return prioritized

    def _convert_to_feature_requests(self, api_endpoint_mapping: Dict, endpoint_params: Dict) -> List[Dict[str, Any]]:
        """
        Convert the endpoint matcher output to feature requests format.
        """
        feature_requests = []
        
        for api_name, endpoints in api_endpoint_mapping.items():
            for endpoint_info in endpoints:
                endpoint_name = endpoint_info['endpoint']
                
                # Get parameters for this endpoint
                params = endpoint_params.get(endpoint_name, {}).get(api_name, {})
                
                # Map parameters to client format
                mapped_params = self._map_to_client_format(api_name, endpoint_name, params)
                
                # Create feature request
                feature_request = {
                    'api': api_name,
                    **mapped_params  # Include all mapped parameters
                }
                
                feature_requests.append(feature_request)
        
        return feature_requests

    def process_features(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Process API requests and return a list of dictionaries containing
        the API used, the request features, and the resulting DataFrame.
        """
        requests = features if isinstance(features, list) else [features]
        outputs: List[Dict[str, Any]] = []

        for feat in requests:
            # Make a copy to safely modify for the client call
            feat_copy = feat.copy()
            api_choice = feat_copy.pop('api', 'polygon')
            client = self.clients.get(api_choice)

            if not client:
                print(f"No client found for API: {api_choice}")
                continue

            try:
                print(f"Fetching data for {api_choice} with features: {feat_copy}")
                raw_pkg = client.fetch_data(feat_copy)
                df = client.parse_response(raw_pkg)

                # Check if df is a tuple (old format) and extract just the DataFrame
                if isinstance(df, tuple) and len(df) > 0:
                    df = df[0]  # Take the first element (DataFrame)
                
                # Ensure df is a DataFrame
                if not isinstance(df, pd.DataFrame):
                    print(f"Warning: parse_response did not return a DataFrame for {api_choice}")
                    df = pd.DataFrame()

                # Append the result dictionary to the outputs list
                outputs.append({
                    'api': api_choice,
                    'features': feat,  # The original, unmodified feature dictionary
                    'df': df
                })
                print(f"Successfully processed request for {api_choice}")
                
            except Exception as e:
                error_msg = str(e)
                if "NOT_FOUND" in error_msg or "Data not found" in error_msg:
                    print(f"No data found for {api_choice} with features {feat_copy}")
                elif "from" in error_msg and "required" in error_msg.lower():
                    print(f"Missing required 'from' parameter for {api_choice} with features {feat_copy}")
                else:
                    print(f"Error processing request for {api_choice} with features {feat_copy}: {e}")
                # Continue with other requests even if one fails
                continue

        return outputs

if __name__ == "__main__":
    config = {
        'polygon_api_key': 'pEP9v2lGpSGlWpMrWdHXqprZsV5MYYbc', 
        'alpha_vantage_api_key': 'WXOG38FYIAUD05SZ'
    }
    ingestor = Ingestor(config)
    
    # Example 1: Traditional feature-based requests
    print("=== Example 1: Traditional Feature Requests ===")
    feature_requests = [
        {
            'api': 'polygon', 
            'ticker': 'AAPL', 
            'multiplier': 1, 
            'timespan': 'day', 
            'from': '2023-01-01', 
            'to': '2023-02-01', 
            'endpoint_type': 0
        },
        {
            'api': 'alpha_vantage', 
            'ticker': 'MSFT', 
            'timespan': 'day', 
            'outputsize': 'compact'
        }
    ]
    
    results = ingestor.process_features(feature_requests)
    print(f"Processed {len(results)} traditional requests.\n")
    
    # Example 2: Natural language processing
    print("=== Example 2: Natural Language Processing ===")
    prompts = [
        "Show me Apple's daily stock price for the last month",
        "Get Microsoft's previous close from Polygon",
        "Show me intraday data for IBM at 5 minute intervals",
        "Get grouped daily aggregates for all stocks on January 15, 2023"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        try:
            nl_results = ingestor.process_natural_language(prompt)
            print(f"Processed {len(nl_results)} requests from natural language.")
            
            for res in nl_results:
                print(f"API: {res['api']}, Features: {res['features']}")
                if not res['df'].empty:
                    print(f"DataFrame Shape: {res['df'].shape}")
                    print(f"DataFrame Head:\n{res['df'].head()}")
                else:
                    print("DataFrame is empty.")
                print("-" * 40)
            
        except Exception as e:
            print(f"Error processing natural language: {e}")