from ollama import chat
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from lmstudio_basemodel import get_schema

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

ollama_model =  'deepseek-r1:32b' #'phi4:latest' #'deepseek-r1:14b' #
openai_model = 'o3-mini'
lmstudio_model = 'mistral-small-24b-instruct-2501'
api = 'lmstudio'

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
lmstudio_client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

API_TYPE = 'lmstudio'
MODEL_NAME = 'mistral-small-24b-instruct-2501'

def llm_chat(prompt: str, schema_name: str):
    """
    Generic chat function that handles both OpenAI and LM Studio APIs.
    
    Args:
        prompt: The prompt to send to the LLM
        schema_name: The name of the schema to use for structured output
    """
    client = openai_client if API_TYPE == "openai" else lmstudio_client
    
    # Get the appropriate schema based on API type
    schema = get_schema(schema_name)
    
    if API_TYPE == "openai":
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    else:  # lmstudio
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format=schema
        )
        return json.loads(response.choices[0].message.content)

def set_api_type(api_type: str):
    """Set the API type to use (openai or lmstudio)"""
    global API_TYPE
    if api_type not in ["openai", "lmstudio"]:
        raise ValueError("API type must be either 'openai' or 'lmstudio'")
    API_TYPE = api_type

def set_model_name(model_name: str):
    """Set the model name to use"""
    global MODEL_NAME
    MODEL_NAME = model_name
