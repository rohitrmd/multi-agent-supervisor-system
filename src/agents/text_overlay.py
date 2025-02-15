from typing import Dict
from ..types.state import AgentState

def create_text_overlay_agent():
    def text_overlay_agent(state: AgentState) -> Dict:
        print("\n✍️ Text Overlay Agent: Processing request...")
        
        # Update state with mock image URL and add a message
        new_state = state.copy()
        new_state["processed_image_url"] = "mock_text_overlay_image.jpg"
        new_state["messages"].append({"role": "system", "content": "Text Overlay Agent: Added text to image"})
        
        return {"state": new_state}
    return text_overlay_agent 