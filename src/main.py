from langgraph.graph import StateGraph, START
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

# Use relative imports (note the . before agents)
from .agents.supervisor import create_supervisor_agent
from .agents.image_generation import create_image_generation_agent
from .agents.text_overlay import create_text_overlay_agent
from .agents.background_removal import create_background_removal_agent
from .agent_types.state import AgentState

def create_workflow():
    # Create the graph
    builder = StateGraph(AgentState)

    # Add nodes for each agent
    builder.add_node("supervisor", create_supervisor_agent())
    builder.add_node("image_generation", create_image_generation_agent())
    builder.add_node("text_overlay", create_text_overlay_agent())
    builder.add_node("background_removal", create_background_removal_agent())

    # Add starting edge
    builder.add_edge(START, "supervisor")

    return builder.compile()

def main():
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        return

    # Create the workflow
    workflow = create_workflow()
    
    # Get user input
    print("\n🤖 Image Processing Multi-Agent System")
    print("----------------------------------------")
    user_instruction = input("\nWhat would you like to do with the image?\n(e.g., 'Generate an image of a sunset and add text to it')\n\nYour request: ")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=user_instruction)],
        "next_agent": None,
        "current_task": None,
        "image_url": None,
        "processed_image_url": None
    }
    
    print("\n🚀 Starting workflow...")
    print("----------------------------------------")
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    # Print results
    print("\n✨ Workflow completed!")
    print("----------------------------------------")
    print("\nExecution path:")
    for msg in final_state["messages"]:
        # Handle both dict messages and Message objects
        content = msg.content if hasattr(msg, 'content') else msg.get('content', str(msg))
        print(f"- {content}")
    
    print(f"\nFinal image URL: {final_state['processed_image_url']}")

if __name__ == "__main__":
    main() 