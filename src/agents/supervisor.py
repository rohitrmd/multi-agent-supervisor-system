from typing import Dict, Literal
from langgraph.types import Command
from ..types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    def supervisor_agent(state: AgentState) -> Command[Literal["image_generation", "text_overlay", "background_removal", "__end__"]]:
        print("\n🎯 Supervisor Agent: Deciding next task...")
        
        # Simple round-robin task assignment for demonstration
        current_task = state["current_task"]
        
        if current_task is None:
            next_agent = "image_generation"
        elif current_task == "image_generation":
            next_agent = "text_overlay"
        elif current_task == "text_overlay":
            next_agent = "background_removal"
        else:
            next_agent = "__end__"  # Using END constant
        
        print(f"➡️ Next agent: {next_agent}")
        
        return Command(
            goto=next_agent,
            update={
                "next_agent": next_agent,
                "current_task": next_agent,
                "messages": state["messages"] + [{"role": "system", "content": f"Supervisor: Routing to {next_agent}"}]
            }
        )
    
    return supervisor_agent 