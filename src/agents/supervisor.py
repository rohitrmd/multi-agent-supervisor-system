from typing import Dict
from langchain_openai import ChatOpenAI
from ..types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    llm = ChatOpenAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE
    )
    
    prompt = """You are a supervisor agent that coordinates image processing tasks.
    Based on the request, determine which tasks need to be executed in sequence.
    Available tasks:
    - image_generation: Generates new images
    - text_overlay: Adds text to images
    - background_removal: Removes background from images
    
    Respond with the next task to execute or 'FINISH' if all tasks are complete.
    Current state of the image and previous tasks will be provided.
    """
    
    def supervisor_agent(state: AgentState) -> Dict:
        # Implementation coming soon
        return {"next_agent": "image_generation", "state": state}
    
    return supervisor_agent 