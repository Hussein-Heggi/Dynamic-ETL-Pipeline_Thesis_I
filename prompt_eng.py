import os
import faiss
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from openai import OpenAI
import torch
import json
import re

api_endpoint_map = {
    "alpha_vantage": {
        "endpoints": [
            "TIME_SERIES_INTRADAY", "TIME_SERIES_DAILY", "TIME_SERIES_WEEKLY", 
            "TIME_SERIES_MONTHLY"
        ],
        "endpoint_descriptions": {
            "TIME_SERIES_INTRADAY": "Intraday stock price data at 1, 5, 15, 30, or 60 minute intervals for short-term trading",
            "TIME_SERIES_DAILY": "Daily historical stock prices with open, high, low, close and volume data",
            "TIME_SERIES_WEEKLY": "Weekly aggregated stock price data for longer-term analysis", 
            "TIME_SERIES_MONTHLY": "Monthly historical stock prices for long-term investment analysis"
        },
        "endpoint_params": {
            "TIME_SERIES_INTRADAY": {
                "required": ["ticker", "timespan"],
                "optional": ["outputsize", "month"]
            },
            "TIME_SERIES_DAILY": {
                "required": ["ticker"],
                "optional": ["outputsize"]
            },
            "TIME_SERIES_WEEKLY": {
                "required": ["ticker"],
                "optional": ["outputsize"]
            },
            "TIME_SERIES_MONTHLY": {
                "required": ["ticker"],
                "optional": ["outputsize"]
            }
        }
    },
    "polygon": {
        "endpoints": [
            "get_aggs", "get_grouped_daily_aggs", "get_daily_open_close_agg", 
            "get_previous_close_agg"
        ],
        "endpoint_descriptions": {
            "get_aggs": "Aggregated stock price bars with custom timeframes and technical indicators",
            "get_grouped_daily_aggs": "Bulk daily stock data for all tickers on specific market dates",
            "get_daily_open_close_agg": "Specific daily opening and closing prices for individual stocks",
            "get_previous_close_agg": "Previous trading day closing prices and market summary data"
        },
        "endpoint_params": {
            "get_aggs": {
                "required": ["ticker"],
                "optional": ["multiplier", "timespan", "from", "to", "adjusted", "sort", "limit"]
            },
            "get_grouped_daily_aggs": {
                "required": ["date"],
                "optional": ["adjusted"]
            },
            "get_daily_open_close_agg": {
                "required": ["ticker", "date"],
                "optional": ["adjusted"]
            },
            "get_previous_close_agg": {
                "required": ["ticker"],
                "optional": ["adjusted"]
            }
        }
    }
}

class endpointMatcher:
    """
    A class that matches user prompts to API endpoints using:
    - GPT for keyword extraction
    - FAISS + FinBERT for semantic similarity
    - Supports API-endpoint mapping structure
    """
    
    def __init__(self, api_endpoint_map, embedding_model="ProsusAI/finbert"):
        """
        Initialize with the API-endpoint mapping structure
        
        Args:
            api_endpoint_map: Dictionary mapping APIs to their endpoints and descriptions
            embedding_model: Transformer model to use
        """
        self.api_endpoint_map = api_endpoint_map
        
        # Build unified endpoint index across all APIs
        self.endpoint_info = self.build_endpoint_index()
        self.endpoint_names = list(self.endpoint_info.keys())
        self.endpoint_texts = [info['description'] for info in self.endpoint_info.values()]
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        
        # Build FAISS index
        self.build_faiss_index()

    def build_endpoint_index(self):
        """
        Create a unified index of all endpoints with their metadata
        Returns dictionary with structure:
        {
            "endpoint_name": {
                "description": "...",
                "apis": ["api1", "api2", ...]
            }
        }
        """
        endpoint_index = {}
        
        for api_name, api_data in self.api_endpoint_map.items():
            for endpoint in api_data['endpoints']:
                if endpoint not in endpoint_index:
                    endpoint_index[endpoint] = {
                        'description': api_data['endpoint_descriptions'][endpoint],
                        'apis': []
                    }
                endpoint_index[endpoint]['apis'].append(api_name)
                
        return endpoint_index

    def generate_embeddings(self, texts):
        """Generate embeddings using transformers model"""
        # Tokenize input
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Perform mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Create attention mask with the same dimensions as token_embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum embeddings along axis 1, but ignore padding tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the mask along axis 1
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Calculate mean
        mean_embeddings = sum_embeddings / sum_mask
        
        # Convert to numpy and normalize
        embeddings = mean_embeddings.numpy()
        embeddings = normalize(embeddings)
        
        return embeddings

    def build_faiss_index(self, index_type="flat", nlist=10):
        """Build FAISS index for endpoint embeddings"""
        self.endpoint_embeddings = self.generate_embeddings(self.endpoint_texts)
        d = self.endpoint_embeddings.shape[1]

        if index_type == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.endpoint_embeddings)
            self.index.nprobe = min(nlist, 10)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.index.add(self.endpoint_embeddings)

    def extract_keywords_with_gpt(self, prompt):
        """Extract relevant keywords from prompt using GPT"""
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable not set.")

        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

        system_prompt = (
            "you are assisting me with extracting keywords from prompts related to finance."
            "Extract relevant data fields or keywords from user requests."
            "Return only a Python list of keywords, e.g., ['keyword1', 'keyword2']."
            "Extract as many key words as possible."
            "if the timeline of such data is implied withing the prompt is implied then output keywords related to it."
        )

        response = client.chat.completions.create(
            model="sonar",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=128,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        # Clean the response more aggressively
        if '[' in content and ']' in content:
            content = content[content.index('['):content.rindex(']')+1]
        try:
            keywords = eval(content)
            if not isinstance(keywords, list):
                keywords = [content]
        except:
            keywords = [word.strip().strip("'\"") 
                    for word in content.strip("[]").split(',')]
        
        return [kw for kw in keywords if kw]  # Remove empty strings

    def semantic_match_faiss(self, phrases):
        """Find matching endpoints using semantic similarity
        Returns dictionary where each endpoint appears only once with its highest score
        """
        phrase_embeddings = self.generate_embeddings(phrases)
        
        # OLD CODE - only picks top 1 match per phrase
        # D, I = self.index.search(phrase_embeddings, k=1)
        
        # NEW CODE - returns multiple matches per keyword (top 3)
        D, I = self.index.search(phrase_embeddings, k=3)
        
        # First pass - collect all matches
        all_matches = {}
        for i, phrase in enumerate(phrases):
            # OLD CODE - only uses the first/best match
            # score = D[i][0]
            # if score > 0.5:  # Similarity threshold
            #     matched_endpoint = self.endpoint_names[I[i][0]]
            #     if matched_endpoint not in all_matches or score > all_matches[matched_endpoint]['score']:
            #         all_matches[matched_endpoint] = {
            #             'phrase': phrase,
            #             'score': round(score, 2),
            #             'apis': self.endpoint_info[matched_endpoint]['apis']
            #         }
            
            # *checks all k matches for this phrase instead of just the first
            for j in range(len(D[i])):  # Iterate through all matches
                score = D[i][j]
                if score > 0.5:  # Similarity threshold
                    matched_endpoint = self.endpoint_names[I[i][j]]
                    if matched_endpoint not in all_matches or score > all_matches[matched_endpoint]['score']:
                        all_matches[matched_endpoint] = {
                            'phrase': phrase,
                            'score': round(score, 2),
                            'apis': self.endpoint_info[matched_endpoint]['apis']
                        }
        
        # Convert to phrase-keyed dictionary (optional)
        matches = {
            match['phrase']: {
                'endpoint': endpoint,
                'score': match['score'],
                'apis': match['apis']
            }
            for endpoint, match in all_matches.items()
        }
        
        return matches

    def match_prompt(self, prompt, target_api=None):
        print(f"\n Prompt: {prompt}")
        
        # Extract keywords
        keywords = self.extract_keywords_with_gpt(prompt)
        print(f" GPT Keywords: {keywords}")
        
        # Get semantic matches
        sem_matches = self.semantic_match_faiss(keywords)
        
        # Filter by target API if specified
        if target_api:
            sem_matches = {
                k: v for k, v in sem_matches.items() 
                if target_api in v['apis']
            }
        
        print(f"\n endpoint Matches: {sem_matches}")
        
        # Create list of recommended APIs
        recommended_apis = set()
        api_endpoint_map = {}
        
        # DEDUPLICATION: Track which endpoints we've already added
        seen_endpoints = set()
        
        for match in sem_matches.values():
            endpoint_name = match['endpoint']
            
            # Skip if we've already processed this endpoint
            if endpoint_name in seen_endpoints:
                continue
                
            seen_endpoints.add(endpoint_name)
            
            for api in match['apis']:
                recommended_apis.add(api)
                if api not in api_endpoint_map:
                    api_endpoint_map[api] = []
                api_endpoint_map[api].append({
                    'endpoint': endpoint_name,
                    'score': match['score']
                })
        
        # Extract parameters for matched endpoints
        endpoint_params = self.endpoint_parameters(keywords, api_endpoint_map)
        print(f"\n Extracted Parameters: {endpoint_params}")
        
        return sem_matches, list(recommended_apis), api_endpoint_map, endpoint_params

    def endpoint_parameters(self, keywords, api_endpoint_map):
        """
        Complete working version with all required methods
        """
        temporal_info = self.extract_temporal_info(keywords)
        
        # Prepare base context without temporal phrases
        base_context = [kw for kw in keywords 
                    if not any(d in kw for d in temporal_info['exact_dates'])]
        
        if temporal_info['description']:
            base_context = [kw for kw in base_context 
                        if temporal_info['description'].lower() not in kw.lower()]
        
        endpoint_params = {}
        
        for api_name, endpoints in api_endpoint_map.items():
            for endpoint_info in endpoints:
                endpoint_name = endpoint_info['endpoint']
                
                try:
                    param_schema = self.api_endpoint_map[api_name]['endpoint_params'][endpoint_name]
                except KeyError:
                    continue
                    
                # Handle all endpoints with standard parameter extraction
                params = self._extract_endpoint_params(
                    endpoint_name,
                    param_schema,
                    base_context,
                    temporal_info,
                    api_name
                )
                if params:
                    endpoint_params.setdefault(endpoint_name, {})[api_name] = params
        
        return endpoint_params

    def _extract_endpoint_params(self, endpoint_name, param_schema, context, temporal_info, api_name):
        """More robust parameter extraction with validation"""
        params = {}
        
        # Get LLM response
        response = self._get_llm_response(endpoint_name, param_schema, " ".join(context), api_name)
        
        if response:
            parsed = self._parse_params_from_response(response, param_schema)
            if parsed:
                # Only include parameters that are in the schema
                valid_params = {}
                for param in param_schema['required'] + param_schema['optional']:
                    if param in parsed:
                        valid_params[param] = parsed[param]
                
                params.update(valid_params)
        
        # Add default parameters for missing required ones
        for req_param in param_schema['required']:
            if req_param not in params:
                # Provide sensible defaults for missing required parameters
                default_values = {
                    'ticker': 'AAPL',  # Default ticker
                    'timespan': 'day',  # Default timespan
                    'date': '2023-01-01',  # Default date
                    'multiplier': 1,  # Default multiplier
                }
                if req_param in default_values:
                    params[req_param] = default_values[req_param]
                    print(f"Added default value for missing required parameter: {req_param} = {default_values[req_param]}")
        
        # Add temporal parameters if needed
        if endpoint_name.startswith('TIME_SERIES'):
            if temporal_info['is_relative'] and 'outputsize' in param_schema.get('optional', []):
                params['outputsize'] = 'compact'
            elif temporal_info['exact_dates']:
                # Use the first exact date found for date parameters
                if endpoint_name in ['get_daily_open_close_agg', 'get_grouped_daily_aggs']:
                    params['date'] = temporal_info['exact_dates'][0]
        
        # Fix for Alpha Vantage timespan parameter
        if api_name == 'alpha_vantage' and endpoint_name == 'TIME_SERIES_INTRADAY':
            # Ensure timespan is one of the valid intervals
            if 'timespan' not in params:
                params['timespan'] = '5min'  # Default to 5min
        
        return params if params else None

    def _get_llm_response(self, endpoint_name, param_schema, context, api_name):
        """
        Get parameter extraction response from LLM
        """
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            return None
            
        client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
        
        system_prompt = (
            f"Extract ONLY JSON parameters for {api_name} {endpoint_name} from: {context}\n"
            f"Required parameters: {param_schema['required']}\n"
            f"Optional parameters: {param_schema['optional']}\n"
            "Return the parameters format according to that stated in each api's documentation.\n"
            "DO NOT include any explanations or text outside the JSON object."
        )
        
        try:
            response = client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=256,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error for {endpoint_name}: {str(e)}")
            return None

    def extract_temporal_info(self, keywords):
        """
        Extract temporal information from keywords
        """
        temporal_info = {
            'description': '',
            'is_relative': False,
            'exact_dates': [],
            'days': None,
            'temporal_phrases': []
        }
        
        date_pattern = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}"
        
        for kw in keywords:
            kw_lower = kw.lower()
            # Check for exact dates
            if re.search(date_pattern, kw, re.IGNORECASE):
                temporal_info['exact_dates'].append(kw)
                temporal_info['temporal_phrases'].append(kw)
            
            # Check for relative time
            elif "last" in kw_lower or "past" in kw_lower:
                match = re.search(r"last\s+(\d+)\s+days", kw_lower)
                if match:
                    temporal_info.update({
                        'description': kw,
                        'is_relative': True,
                        'temporal_phrases': [kw],
                        'days': int(match.group(1))
                    })
        
        return temporal_info

    def _parse_params_from_response(self, response, param_schema):
        """More robust JSON parsing with multiple fallback strategies"""
        if not response:
            return None
            
        try:
            # First try direct JSON parse
            return json.loads(response)
        except json.JSONDecodeError:
            try:
                # Try extracting JSON from markdown code blocks
                if '```json' in response:
                    json_str = response.split('```json')[1].split('```')[0]
                    return json.loads(json_str)
                elif '```' in response:
                    json_str = response.split('```')[1].split('```')[0]
                    return json.loads(json_str)
                    
                # Try extracting just the first {...} block
                if '{' in response and '}' in response:
                    json_str = response[response.index('{'):response.rindex('}')+1]
                    return json.loads(json_str)
                    
                # Final fallback - try to construct basic JSON from key-value pairs
                params = {}
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                for line in lines:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        params[key.strip()] = val.strip().strip('"\'')
                return params if params else None
                
            except Exception as e:
                print(f"Failed to parse response after multiple attempts: {str(e)}")
                return None

if __name__ == "__main__":
    matcher = endpointMatcher(api_endpoint_map, embedding_model="ProsusAI/finbert")

    prompt = "For Apple (AAPL), show me its most recent previous close using get_previous_close_agg, its open and close prices on January 15, 2023 using get_daily_open_close_agg, grouped daily aggregate data for all U.S. stocks on that same date using get_grouped_daily_aggs, the latest trade and last quote for Apple, a full daily FX series against USD, the weekly and monthly time series, the daily VWAP, and also a snapshot of Apple in the U.S. stock market."
    endpoint_matches, recommended_apis, api_endpoint_mapping, endpoint_params = matcher.match_prompt(prompt)
    
    # Print results in the requested format
    print("\n=== Recommended APIs ===")
    for api in recommended_apis:
        print(f"- {api}")
    
    print("\n=== Recommended Endpoints for Each API ===")
    for api, endpoints in api_endpoint_mapping.items():
        print(f"\nAPI: {api}")
        for endpoint in endpoints:
            print(f"  - {endpoint['endpoint']} (confidence score: {endpoint['score']})")
    
    print("\n=== Recommended Parameters for Each Endpoint ===")
    if endpoint_params:
        for endpoint, params in endpoint_params.items():
            print(f"\nEndpoint: {endpoint}")
            for api, param_values in params.items():
                print(f"  API: {api}")
                if param_values:  # Only print if parameters exist
                    for param, value in param_values.items():
                        print(f"    - {param}: {value}")
                else:
                    print("    - No parameters extracted")
    else:
        print("No parameters were extracted for any endpoints")