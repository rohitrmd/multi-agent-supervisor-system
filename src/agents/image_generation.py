from typing import Dict, Literal
from langgraph.types import Command
from ..agent_types.state import AgentState

def create_image_generation_agent():
    def image_generation_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n🎨 Image Generation Agent: Processing request...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_generated_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "Image Generation Agent: Generated new image"}
                ]
            }
        )
    
    return image_generation_agent 