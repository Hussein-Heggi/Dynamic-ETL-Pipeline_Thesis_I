import os
from typing import cast
from openai import OpenAI


# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_llm_recipe(user_keywords: list, allowed_features_prompt: str) -> str:
    """
    Calls the OpenAI API to translate user keywords into a DSL JSON recipe.
    """

    system_prompt = f"""
You are an expert financial data analyst that converts a list of keywords into a JSON recipe.

**Your output MUST follow this exact format:**
Each feature object in the JSON list MUST have a "name" key and a "params" key.

EXAMPLE:
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

RULES:
1. You MUST only output a valid JSON object. Do not include any other text or explanations.
2. For parameters that specify a column (like "on", "high", "low", "close"), the value MUST be a string literal of the column name.
3. All window, period, or standard deviation values MUST be integers, not strings.

ALLOWED FEATURES:
{allowed_features_prompt}
"""

    # the specific task for this run
    user_prompt = f"""
KEY FEATURES LIST:
{user_keywords}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
