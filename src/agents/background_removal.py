from typing import Dict
from ..types.state import AgentState

def create_background_removal_agent():
    def background_removal_agent(state: AgentState) -> Dict:
        print("\n✂️ Background Removal Agent: Processing request...")
        
        # Update state with mock image URL and add a message
        new_state = state.copy()
        new_state["processed_image_url"] = "mock_bg_removed_image.jpg"
        new_state["messages"].append({"role": "system", "content": "Background Removal Agent: Removed image background"})
        
        return {"state": new_state}
    return background_removal_agent 