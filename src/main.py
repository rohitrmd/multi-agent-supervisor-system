from langgraph.graph import Graph
from langchain_core.messages import HumanMessage
from typing import Dict

from .agents.supervisor import create_supervisor_agent
from .agents.image_generation import create_image_generation_agent
from .agents.text_overlay import create_text_overlay_agent
from .agents.background_removal import create_background_removal_agent
from .types.state import AgentState

def create_workflow():
    workflow = Graph()

    # Add nodes
    workflow.add_node("supervisor", create_supervisor_agent())
    workflow.add_node("image_generation", create_image_generation_agent())
    workflow.add_node("text_overlay", create_text_overlay_agent())
    workflow.add_node("background_removal", create_background_removal_agent())

    # Add edges based on supervisor decisions
    def route_by_next_agent(state: AgentState) -> str:
        return state["next_agent"] if state["next_agent"] else "END"

    workflow.add_edge("supervisor", route_by_next_agent)
    workflow.add_edge("image_generation", "supervisor")
    workflow.add_edge("text_overlay", "supervisor")
    workflow.add_edge("background_removal", "supervisor")

    return workflow.compile()

def main():
    workflow = create_workflow()
    
    # Get user input
    user_instruction = input("Please enter your image processing instruction: ")
    
    # Initialize state
    initial_state = AgentState(
        messages=[HumanMessage(content=user_instruction)],
        next_agent="supervisor",
        current_task=None,
        image_url=None,
        processed_image_url=None
    )
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    # Print results
    print("\nProcessing completed!")
    print(f"Final image URL: {final_state['processed_image_url']}")
    print("\nExecution path:")
    for msg in final_state["messages"]:
        print(f"- {msg.content}")

if __name__ == "__main__":
    main() 