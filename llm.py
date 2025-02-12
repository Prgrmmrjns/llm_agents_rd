import json
from openai import OpenAI
from models import ValidationList, SearchKeywords, ReformulatedOptions

lmstudio_model = 'mistral-small-24b-instruct-2501'

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

API_TYPE = 'lmstudio'
MODEL_NAME = 'mistral-small-24b-instruct-2501'

def get_schema(schema_name: str):
    """Get the appropriate schema based on schema name and API type."""
    schema_map = {
        "validation_list": ValidationList.model_json_schema(),
        "search": SearchKeywords.model_json_schema(),
        "reformulated_options": ReformulatedOptions.model_json_schema()
    }
    
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_map.get(schema_name)
        }
    }

def llm_chat(prompt: str, schema_name: str):
    """
    Generic chat function that handles both OpenAI and LM Studio APIs.
    
    Args:
        prompt: The prompt to send to the LLM
        schema_name: The name of the schema to use for structured output
    """
    
    # Get the appropriate schema based on API type
    schema = get_schema(schema_name)
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema
    )
    return json.loads(response.choices[0].message.content)
