from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from ..types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    llm = ChatOpenAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE
    )
    
    system_prompt = """You are a supervisor agent coordinating image processing tasks.
    Based on the user's request, determine which task should be executed next.
    
    Available tasks:
    1. image_generation - When user needs to create a new image
    2. text_overlay - When text needs to be added to an image
    3. background_removal - When background needs to be removed from an image
    
    Rules:
    - If the request mentions creating/generating an image, start with 'image_generation'
    - If the request mentions adding text/caption, use 'text_overlay'
    - If the request mentions removing/deleting background, use 'background_removal'
    - If all requested tasks are complete, respond with '__end__'
    - Always process one task at a time in logical order
    
    Respond with the next task to execute based on the current state and user request.
    """

    def supervisor_agent(state: AgentState) -> Command[Literal["image_generation", "text_overlay", "background_removal", "__end__"]]:
        print("\n🎯 Supervisor Agent: Deciding next task...")
        
        # Get the initial request if this is the first run
        if len(state["messages"]) == 1:
            user_request = state["messages"][0].content
        else:
            # Get the last message to see what was just completed
            user_request = state["messages"][-1].content

        # Use LLM to decide next task
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            User Request: {state["messages"][0].content}
            Current Task: {state["current_task"]}
            Previous Tasks Completed: {[msg.content for msg in state["messages"] if msg.name in ["image_generation", "text_overlay", "background_removal"]]}
            
            What should be the next task?
            """)
        ]
        
        response = llm.invoke(messages).content
        
        # Parse the response to get the next task
        if "image_generation" in response.lower():
            next_agent = "image_generation"
        elif "text_overlay" in response.lower():
            next_agent = "text_overlay"
        elif "background_removal" in response.lower():
            next_agent = "background_removal"
        else:
            next_agent = "__end__"
        
        print(f"➡️ Next agent: {next_agent}")
        
        return Command(
            goto=next_agent,
            update={
                "next_agent": next_agent,
                "current_task": next_agent,
                "messages": state["messages"] + [
                    SystemMessage(content=f"Supervisor: Routing to {next_agent}")
                ]
            }
        )
    
    return supervisor_agent 