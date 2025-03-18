from typing import Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import json

# Initialize judge LLM
judge_llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

async def evaluate_task_completion(run, example) -> Dict:
    """
    Uses GPT-4 to evaluate if the workflow completed the requested tasks correctly.
    """
    try:
        # Extract actual sequence and messages
        actual_sequence = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Get expected sequence
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
        explanation = response.content.split("\n", 1)[1] if "\n" in response.content else ""
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": f"""
                Judge's Evaluation:
                {response.content}

                Expected Sequence:
                {json.dumps(expected_sequence, indent=2)}

                Actual Sequence:
                {json.dumps(actual_sequence, indent=2)}
            """
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}"
        }

async def check_node_execution(run, example) -> Dict:
    """
    Uses GPT-4 to evaluate if the correct nodes/agents were executed in the workflow.
    """
    try:
        # Initialize the judge LLM
        judge_llm = ChatOpenAI(model="gpt-4", temperature=0)
        
        # Extract all agent messages
        agent_messages = [
            msg["content"] for msg in run.outputs["messages"] 
            if msg.get("role") == "system" and "Agent:" in msg.get("content", "")
        ]
        
        # Extract unique agent names from messages
        executed_agents = set(
            msg.split("Agent:")[0].strip() 
            for msg in agent_messages 
            if "Agent:" in msg
        )
        
        # Prepare instructions for the judge
        instructions = """
        You are an evaluation judge analyzing workflow execution. Given the sequence of agent actions 
        and the expected workflow steps, determine if:
        
        1. All necessary agents were involved
        2. No unexpected agents were used
        3. The agents executed their tasks appropriately
        
        Respond with either 'CORRECT' or 'INCORRECT', followed by a brief analysis of the agent execution.
        """
        
        # Prepare the comparison message
        comparison_msg = f"""
        EXPECTED WORKFLOW:
        {json.dumps(example.outputs["expected_sequence"], indent=2)}
        
        ACTUAL AGENT EXECUTIONS:
        {json.dumps(agent_messages, indent=2)}
        
        Agents Involved: {", ".join(executed_agents)}
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
        explanation = response.content.split("\n", 1)[1] if "\n" in response.content else ""
        
        return {
            "score": 1.0 if is_correct else 0.0,
            "reasoning": f"""
                Judge's Analysis:
                {response.content}

                Agents Involved:
                {json.dumps(list(executed_agents), indent=2)}

                Full Execution Sequence:
                {json.dumps(agent_messages, indent=2)}
            """
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