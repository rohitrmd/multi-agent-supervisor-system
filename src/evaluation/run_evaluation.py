from langsmith import Client
from dotenv import load_dotenv
import os
import asyncio
import pandas as pd
from tabulate import tabulate
import json
from datetime import datetime

from ..main import create_workflow
from .evaluators import evaluate_task_completion
from .create_dataset import create_evaluation_dataset

async def run_evaluations():
    # Initialize environment and check API key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
        
    print("\n🚀 Starting Evaluation Process")
    print("==============================")
    
    # Step 1: Dataset Creation/Retrieval
    print("\n1️⃣ Setting up test dataset...")
    dataset = create_evaluation_dataset()
    client = Client()
    print("✓ Dataset ready with test case: Generate image with text overlay")
    
    # Step 2: Workflow Setup
    print("\n2️⃣ Initializing workflow...")
    workflow = create_workflow()
    print("✓ Multi-agent workflow initialized")
    
    # Step 3: Input Preparation
    print("\n3️⃣ Preparing input processor...")
    def process_request(inputs: dict) -> dict:
        return {
            "messages": [
                {"role": "user", "content": inputs["request"]}
            ],
            "next_agent": None,
            "current_task": None,
            "image_url": None,
            "processed_image_url": None
        }
    print("✓ Input processor ready")
    
    # Step 4: Evaluation Setup
    print("\n4️⃣ Setting up evaluation...")
    target = process_request | workflow
    print("✓ Evaluation target configured")
    
    # Step 5: Run Evaluation
    print("\n5️⃣ Running evaluation...")
    print("   • Executing workflow")
    print("   • Checking task completion")
    experiment_results = await client.aevaluate(
        target,
        data=dataset.name,
        evaluators=[evaluate_task_completion],
        experiment_prefix="image_processing_eval",
        num_repetitions=1,
        max_concurrency=1
    )
    print("✓ Evaluation complete")
    
    # Step 6: Process Results
    print("\n6️⃣ Processing results...")
    results_df = experiment_results.to_pandas()
    
    results_dict = {
        "Test Request": {
            "input": results_df['inputs.request'].iloc[0],
            "expected_sequence": results_df['reference.expected_sequence'].iloc[0]
        },
        "Execution Results": {
            "agent_messages": results_df['outputs.messages'].iloc[0],
            "final_state": {
                "next_agent": results_df['outputs.next_agent'].iloc[0],
                "current_task": results_df['outputs.current_task'].iloc[0],
                "image_url": results_df['outputs.image_url'].iloc[0],
                "processed_image_url": results_df['outputs.processed_image_url'].iloc[0]
            }
        },
        "Evaluation": {
            "task_completion_score": float(results_df['feedback.evaluate_task_completion'].iloc[0]),
            "execution_time_seconds": float(results_df['execution_time'].iloc[0])
        }
    }
    
    # Step 7: Display Results
    print("\n7️⃣ Evaluation Results")
    print("===================")
    print("\nDetailed Results:")
    print(json.dumps(results_dict, indent=2, ensure_ascii=False))
    
    # Step 8: Summary
    print("\n8️⃣ Quick Summary")
    print("===============")
    print(f"• Request: {results_dict['Test Request']['input']}")
    print(f"• Task Completion Score: {results_dict['Evaluation']['task_completion_score']}")
    print(f"• Execution Time: {results_dict['Evaluation']['execution_time_seconds']:.2f} seconds")
    print(f"• Final State: {results_dict['Execution Results']['final_state']['next_agent']}")
    
    return experiment_results

if __name__ == "__main__":
    asyncio.run(run_evaluations()) 