from smolagents import (
    CodeAgent,
    HfApiModel,
    LiteLLMModel,
    ToolCallingAgent,
    tool
)
import pandas as pd


# Then we run the agentic part!
model = LiteLLMModel(
    model_id="ollama_chat/openthinker:latest",
    api_base="http://localhost:11434",  # replace with remote open-ai compatible server if necessary
    api_key="your-api-key",  # replace with API key if necessary
    num_ctx=8192,  # ollama default is 2048 which will often fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

rd_info = pd.read_csv('csv/phenotype_data.csv')
filtered_rd_info = rd_info[rd_info['Name'] == 'Vascular Ehlers-Danlos syndrome']
@tool
def get_rare_disease_info(rare_disease: str) -> str:
    """
    Get information about a rare disease from the database.
    
    Args:
        rare_disease: the rare disease to get information about
    Returns:
        A string containing the information about the rare disease.
    """
    return filtered_rd_info.to_string()

search_agent = ToolCallingAgent(
    tools=[get_rare_disease_info],
    model=model,
    name="search_agent",
    description="This is an agent that can get information about a rare disease from the database.",
)

manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[search_agent],
)
manager_agent.run("Tell me about the disease Vascular Ehlers-Danlos Syndrome")