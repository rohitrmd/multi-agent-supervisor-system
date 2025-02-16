# Multi-Agent Image Processing System

A LangGraph-based system that coordinates multiple AI agents for image processing tasks using the Agent-Supervisor pattern.

## Overview

This project implements a multi-agent system based on LangGraph's Agent-Supervisor pattern, where a supervisor agent coordinates multiple specialized image processing agents. The system demonstrates the use of LangGraph's edgeless graph architecture and Command construct for agent coordination.

### System Architecture

1. **Supervisor Agent**
   - Coordinates the workflow
   - Makes intelligent decisions about task sequencing
   - Routes requests to appropriate agents using LangGraph's Command construct

2. **Task Agents**
   - Image Generation Agent: Handles image creation requests
   - Text Overlay Agent: Adds text to images
   - Background Removal Agent: Removes image backgrounds

### Key LangGraph Features Used

1. **Edgeless Graph Architecture**
   - Instead of explicit edges between nodes, routing is handled by agent Commands
   - Each agent returns a Command that specifies the next agent to run
   - Simplifies graph structure and makes it more flexible

2. **Command Construct**
   ```python
   Command(
       goto="next_agent",
       update={
           "next_agent": "next_agent",
           "current_task": "current_task",
           "messages": [...],
       }
   )
   ```
   - `goto`: Specifies the next agent to execute
   - `update`: Updates the state that's passed between agents

3. **StateGraph**
   ```python
   builder = StateGraph(AgentState)
   builder.add_node("supervisor", create_supervisor_agent())
   builder.add_edge(START, "supervisor")
   ```
   - Manages state transitions between agents
   - Only requires initial edge from START to supervisor

## Setup

1. Create a virtual environment:
