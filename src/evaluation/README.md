# Multi-Agent System Evaluation Framework

This framework implements a comprehensive evaluation strategy for multi-agent systems using three distinct criteria.

## Evaluation Criteria

### 1. Task Completion Evaluation
Assesses if the entire multi-agent system successfully completed all requested tasks.
- Verifies all required tasks were completed
- Checks task execution order
- Validates final output against requirements

### 2. Node Execution Path Analysis
Examines the interaction patterns and execution sequence of agents.
- Confirms all necessary agents were involved
- Validates execution order
- Identifies any unnecessary agent invocations

### 3. Individual Node Evaluation
Focuses on specific agent performance (example: Image Generation Agent).
- Verifies individual agent execution
- Checks specific agent functionality
- Provides targeted performance insights

## Running Evaluations

From the project root directory, run:

```bash
python -m src.evaluation.run_evaluation
```

## Output Format

The evaluation provides detailed scores and reasoning for each criterion:

```
Evaluation Results by Criteria
==============================

1️⃣ Task Completion Evaluation:
   Score: 1.0
   Analysis: All tasks completed successfully...

2️⃣ Node Execution Analysis:
   Score: 1.0
   Analysis: Agents executed in correct sequence...

3️⃣ Image Generation Node Check:
   Score: 1.0
   Analysis: Image Generation Agent called successfully...
```

## Implementation Details

- Uses GPT-4 as evaluation judge
- Provides scores from 0.0 to 1.0
- Includes detailed reasoning for each score
- Stores results in LangSmith for tracking

## Components

1. **Test Dataset** (`create_dataset.py`)
   - Predefined test cases with expected outcomes
   - Stored in LangSmith for tracking

2. **Evaluators** (`evaluators.py`)
   - GPT-4 powered evaluation functions
   - Task completion checker
   - Node execution analyzer

3. **Runner** (`run_evaluation.py`)
   - Main evaluation script
   - Results processing and display
   - LangSmith integration

## Project Structure
```
evaluation/
├── README.md           # This file
├── evaluators.py       # Evaluation functions
├── create_dataset.py   # Test dataset creation
└── run_evaluation.py   # Main evaluation script
``` 