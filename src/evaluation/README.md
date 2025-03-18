# Evaluation Framework

This framework assesses the performance and correctness of the multi-agent workflow using GPT-4 as a judge.

## Running Evaluations

From the project root directory, run:

```bash
python -m src.evaluation.run_evaluation
```

## What Gets Evaluated

1. **Task Completion**
   - Whether all requested tasks were completed correctly
   - If tasks were executed in logical order
   - Quality of task execution

2. **Node Execution**
   - Whether the right agents were involved
   - If agent interactions made sense
   - Efficiency of the workflow

## Example Output
```
🚀 Starting Evaluation Process
==============================

1️⃣ Setting up test dataset...
✓ Dataset ready with test case: Generate image with text overlay

[... evaluation steps ...]

8️⃣ Quick Summary
===============
• Request: Generate an image of a sunset and add text 'Beautiful Evening'
• Task Completion Score: 1.0
• Node Execution Score: 1.0
• Execution Time: 2.27 seconds
```

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