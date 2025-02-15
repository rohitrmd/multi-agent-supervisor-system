from typing import Dict
from ..types.state import AgentState

def create_image_generation_agent():
    def image_generation_agent(state: AgentState) -> Dict:
        print("\n🎨 Image Generation Agent: Processing request...")
        
        # Update state with mock image URL and add a message
        new_state = state.copy()
        new_state["processed_image_url"] = "mock_generated_image.jpg"
        new_state["messages"].append({"role": "system", "content": "Image Generation Agent: Generated new image"})
        
        return {"state": new_state}
    return image_generation_agent 