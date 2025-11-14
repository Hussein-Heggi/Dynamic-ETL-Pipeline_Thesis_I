import os
from typing import cast

from httpx import URL
from openai import OpenAI

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if os.getenv("OPENAI_API_BASE_URL"):
    client.base_url = URL(cast(str, os.getenv("OPENAI_API_BASE_URL")))

model = (
    os.getenv("OPENAI_API_MODEL") if os.getenv("OPENAI_API_MODEL") else "gpt-4o-mini"
)


def get_llm_recipe(
    user_keywords: list[str],
    allowed_features_prompt: str,
    available_columns: list[str],
) -> str:
    """
    Calls the OpenAI API to translate user keywords into a DSL JSON recipe.

    Args:
        user_keywords: List of feature keywords from the user
        allowed_features_prompt: Formatted description of available features
        available_columns: List of column names available in the DataFrame
    """

    system_prompt = f"""
You are an expert financial data analyst that converts a list of keywords into a JSON recipe.

**Your output MUST follow this exact format:**
Each feature object in the JSON list MUST have a "name" key and a "params" key.

EXAMPLE 1 - Standard Feature:
INPUT KEYWORDS: ["20 day sma on close"]
OUTPUT JSON:
{{
  "features": [
    {{
      "name": "sma",
      "params": {{
        "on": "close",
        "window": 20
      }}
    }}
  ]
}}

EXAMPLE 2 - Mix of Standard and Custom Features:
INPUT KEYWORDS: ["14 day rsi", "price to 20-day high ratio", "normalized momentum indicator"]
OUTPUT JSON:
{{
  "features": [
    {{
      "name": "rsi",
      "params": {{
        "on": "close",
        "window": 14
      }}
    }},
    {{
      "name": "custom_price_high_ratio",
      "params": {{
        "code": "series = g['close'] / g['high'].rolling(20).max()",
        "as": "price_high_ratio_20"
      }}
    }},
    {{
      "name": "custom_norm_momentum",
      "params": {{
        "code": "momentum = g['close'] - g['close'].shift(10)\\nrolling_std = g['close'].rolling(10).std()\\nseries = momentum / rolling_std",
        "as": "normalized_momentum_10"
      }}
    }}
  ]
}}

RULES:
1. You MUST only output a valid JSON object. Do not include any other text or explanations.
2. For parameters that specify a column (like "on", "high", "low", "close"), the value MUST be a string literal of the column name.
3. All window, period, or standard deviation values MUST be integers, not strings.

CUSTOM FEATURES:
- If a requested feature is NOT in the allowed features list, you can create a custom feature
- Custom feature names MUST start with the prefix "custom_" (e.g., "custom_price_ratio", "custom_my_indicator")
- Custom features require two parameters:
  * "code": Python code string that computes the feature
  * "as": The output column name for the feature
- The code runs in a RESTRICTED ENVIRONMENT with:
  * Safe builtins only (NO file I/O, NO imports, NO exec/eval, NO system calls, NO print)
  * Available modules: numpy (as np), pandas (as pd), math, random
  * Input variable 'g': The group DataFrame (one ticker's data)
  * Output: MUST assign the result to a variable named 'series' (must be a pd.Series)
- For multiline code, use \\n to separate lines in the JSON string
- Example code patterns:
  * "series = g['close'] / g['open']"
  * "series = (g['high'] + g['low']) / 2"
  * "series = g['close'].rolling(10).mean() / g['close']"
  * "series = np.log(g['close'] / g['close'].shift(1))"
  * "momentum = g['close'].diff(5)\\nseries = momentum / g['close'].rolling(20).std()"

ALLOWED FEATURES:
{allowed_features_prompt}

AVAILABLE DATAFRAME COLUMNS:
The DataFrame you are working with has the following columns available:
{', '.join(available_columns)}

When specifying column parameters (like "on", "high", "low", "close", "volume", etc.),
you MUST use only the columns listed above.
"""

    # the specific task for this run
    user_prompt = f"""
KEY FEATURES LIST:
{user_keywords}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # Set temperature to 0 for maximum consistency
        )

        return cast(str, response.choices[0].message.content)

    except Exception as e:
        print(f"An error occurred while calling the OpenAI API: {e}")
        return "{}"
