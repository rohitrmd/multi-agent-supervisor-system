from typing import Dict, Literal
from langgraph.types import Command
from ..types.state import AgentState

def create_background_removal_agent():
    def background_removal_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n✂️ Background Removal Agent: Processing request...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_bg_removed_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "Background Removal Agent: Removed image background"}
                ]
            }
        )
    
    return background_removal_agent 