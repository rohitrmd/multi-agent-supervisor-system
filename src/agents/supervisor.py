from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command

# Simplified imports without src
from ..agent_types.state import AgentState
from ..config.settings import SUPERVISOR_MODEL, SUPERVISOR_TEMPERATURE

def create_supervisor_agent():
    llm = ChatOpenAI(
        model=SUPERVISOR_MODEL,
        temperature=SUPERVISOR_TEMPERATURE
    )
    
    system_prompt = """You are a supervisor agent coordinating image processing tasks.
    Based on the user's request and current state, determine which task should be executed next.
    
    Available tasks:
    1. image_generation - When user needs to create a new image
    2. text_overlay - When text needs to be added to an image
    3. background_removal - When background needs to be removed from an image
    
    Rules:
    - Process tasks in sequence until all requested operations are complete
    - If the request mentions creating/generating an image, start with 'image_generation'
    - After image generation, if text/caption is requested, use 'text_overlay'
    - If the request mentions removing/deleting background, use 'background_removal'
    - Only respond with '__end__' when all requested tasks are complete
    - Consider both the original request and the current task state when deciding the next task
    
    Example sequences:
    - "Generate an image and add text" → image_generation → text_overlay → __end__
    - "Create an image, remove background, add text" → image_generation → background_removal → text_overlay → __end__
    """

    def supervisor_agent(state: AgentState) -> Command[Literal["image_generation", "text_overlay", "background_removal", "__end__"]]:
        print("\n🎯 Supervisor Agent: Deciding next task...")
        
        # Get the initial request if this is the first run
        messages = state["messages"]
        user_request = messages[0]["content"] if isinstance(messages[0], dict) else messages[0].content
        
        # Use LLM to decide next task
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Original Request: {user_request}
            Current Task: {state["current_task"]}
            
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
                    {"role": "system", "content": f"Supervisor: Routing to {next_agent}"}
                ]
            }
        )
    
    return supervisor_agent 