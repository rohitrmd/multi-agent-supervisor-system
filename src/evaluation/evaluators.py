from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langsmith.schemas import Run, Example
import json

# Initialize judge LLM
judge_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

async def evaluate_task_completion(run: Run, example: Example) -> Dict:
    """
    Uses GPT-4 to evaluate if the workflow completed the requested tasks correctly.
    
    Args:
        run: Execution trace containing actual outputs
        example: Test case containing expected behavior
    """
    try:
        # Extract actual sequence from run outputs
        actual_sequence = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Get expected sequence from example
        expected_sequence = example.outputs["expected_sequence"]
        
        # Prepare instructions for the judge
        instructions = """
        You are an evaluation judge. Given the actual sequence of agent actions and the expected sequence,
        determine if the workflow completed all required tasks correctly.
        
        Consider:
        1. Were all expected actions performed?
        2. Were they performed in a logical order?
        3. Did any unexpected or unnecessary actions occur?
        
        Respond with either 'CORRECT' or 'INCORRECT', followed by a brief explanation.
        """
        
        # Prepare the comparison message
        comparison_msg = f"""
        Original Request: {run.inputs.get('request', 'No request found')}
        
        EXPECTED SEQUENCE:
        {json.dumps(expected_sequence, indent=2)}
        
        ACTUAL SEQUENCE:
        {json.dumps(actual_sequence, indent=2)}
        """
        
        # Get judge's evaluation
        response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comparison_msg}
            ]
        )
        
        # Parse the response
        is_correct = response.content.upper().startswith("CORRECT")
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": response.content
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }

async def check_node_execution(run: Run, example: Example) -> Dict:
    """
    Uses GPT-4 to evaluate if the correct nodes/agents were executed in the workflow.
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Extract agent messages and their sequence
        agent_messages = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Extract agent sequence from messages
        agent_sequence = [
            msg.split("Agent:")[0].strip() 
            for msg in agent_messages
        ]
        
        instructions = """
        You are an evaluation judge analyzing workflow execution. Given the sequence of agent actions,
        determine if:
        
        1. All necessary agents were involved based on the request
        2. The agents executed their tasks in the correct order
        3. The sequence makes logical sense for the task
        
        Respond with either 'CORRECT' or 'INCORRECT', followed by a brief analysis.
        """
        
        comparison_msg = f"""
        Original Request: {run.inputs.get('request', 'No request found')}
        
        EXPECTED WORKFLOW:
        {json.dumps(example.outputs["expected_sequence"], indent=2)}
        
        ACTUAL EXECUTIONS:
        {json.dumps(agent_messages, indent=2)}
        
        Agent Sequence: {json.dumps(agent_sequence, indent=2)}
        """
        
        response = await judge_llm.ainvoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": comparison_msg}
            ]
        )
        
        is_correct = response.content.upper().startswith("CORRECT")
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": response.content
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }

async def check_state_transitions(run, example) -> Dict:
    """
    Analyzes the state transitions between agents.
    """
    try:
        transitions = []
        current_task = None
        
        for msg in run.outputs.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "system" and "Agent:" in msg.get("content", ""):
                new_task = msg["content"].split("Agent:")[0].strip()
                if current_task:
                    transitions.append(f"{current_task} -> {new_task}")
                current_task = new_task
        
        return {
            "score": float(len(transitions) > 0),
            "reasoning": f"State transitions: {transitions}"
        }
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error in evaluation: {str(e)}"
        } 