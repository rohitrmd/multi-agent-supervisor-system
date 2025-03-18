"""
Multi-Agent System Evaluation Framework

This module implements three key evaluation criteria:
1. Task Completion: Evaluates if the entire multi-agent system completed the requested tasks correctly
2. Node Execution Path: Analyzes if agents were executed in the correct sequence
3. Individual Node Execution: Checks specific node/agent performance

Each evaluator returns a score (0.0-1.0) and detailed reasoning.
"""

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
    Evaluation Criteria 1: Task Completion
    
    Evaluates if the multi-agent system as a whole completed all requested tasks correctly.
    Considers:
    - All required tasks were completed
    - Tasks were done in logical order
    - Final output matches requirements
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
    Evaluation Criteria 2: Node Execution Path
    
    Analyzes the sequence of agent executions to verify correct workflow.
    Checks:
    - All necessary agents were involved
    - Agents executed in correct order
    - No unnecessary agent invocations
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


async def check_image_generation_node(run: Run, example: Example) -> Dict:
    """
    Evaluation Criteria 3: Individual Node Execution
    
    Example of individual node evaluation, focusing on Image Generation Agent.
    Verifies:
    - If the specific agent was called
    - Simple binary check of agent involvement
    """
    try:
        judge_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Extract messages specifically from Image Generation Agent
        image_gen_messages = [
            msg["content"] for msg in run.outputs.get("messages", [])
            if msg.get("role") == "system" and "Image Generation Agent:" in msg.get("content", "")
        ]
        
        instructions = """
        You are an evaluation judge. Your only task is to check if the Image Generation Agent was called.
        
        Respond with:
        - 'CORRECT' if you see any messages from the Image Generation Agent
        - 'INCORRECT' if there are no messages from the Image Generation Agent
        """
        
        comparison_msg = f"""
        Messages from Image Generation Agent:
        {json.dumps(image_gen_messages, indent=2)}
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