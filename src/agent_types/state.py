from typing import List, TypedDict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    next_agent: Optional[str]
    current_task: Optional[str]
    image_url: Optional[str]
    processed_image_url: Optional[str] 