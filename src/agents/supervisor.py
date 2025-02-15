from typing import Dict
from langchain_openai import ChatOpenAI
from ..types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    def supervisor_agent(state: AgentState) -> Dict:
        print("\n🎯 Supervisor Agent: Deciding next task...")
        
        # Simple round-robin task assignment for demonstration
        current_task = state["current_task"]
        new_state = state.copy()
        
        if current_task is None:
            next_agent = "image_generation"
        elif current_task == "image_generation":
            next_agent = "text_overlay"
        elif current_task == "text_overlay":
            next_agent = "background_removal"
        else:
            next_agent = None  # This will end the workflow
        
        new_state["next_agent"] = next_agent
        new_state["current_task"] = next_agent
        new_state["messages"].append({"role": "system", "content": f"Supervisor: Routing to {next_agent if next_agent else 'END'}"})
        
        print(f"➡️ Next agent: {next_agent if next_agent else 'END'}")
        return {"next_agent": next_agent, "state": new_state}
    
    return supervisor_agent 