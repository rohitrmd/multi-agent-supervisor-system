from typing import Dict, Literal
from langgraph.types import Command
from ..types.state import AgentState

def create_text_overlay_agent():
    def text_overlay_agent(state: AgentState) -> Command[Literal["supervisor"]]:
        print("\n✍️ Text Overlay Agent: Processing request...")
        
        return Command(
            goto="supervisor",
            update={
                "processed_image_url": "mock_text_overlay_image.jpg",
                "messages": state["messages"] + [
                    {"role": "system", "content": "Text Overlay Agent: Added text to image"}
                ]
            }
        )
    
    return text_overlay_agent 