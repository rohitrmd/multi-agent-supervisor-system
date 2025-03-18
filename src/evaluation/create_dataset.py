from langsmith import Client
from langsmith.utils import LangSmithConflictError

def create_evaluation_dataset():
    client = Client()
    dataset_name = "image_processing_agent"
    
    # Just one test case
    test_cases = [
        {
            "request": "Generate an image of a sunset and add 'Beautiful Evening' text",
            "expected_sequence": [
                "Image Generation Agent: Generated new image",
                "Text Overlay Agent: Added text to image"
            ]
        }
    ]
    
    try:
        # Delete existing dataset if it exists
        existing_datasets = client.list_datasets()
        for dataset in existing_datasets:
            if dataset.name == dataset_name:
                client.delete_dataset(dataset_id=dataset.id)
                print(f"Deleted existing dataset: {dataset_name}")
        
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Test cases for multi-agent image processing system"
        )
        
        # Add examples to the dataset
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[{"request": case["request"]} for case in test_cases],
            outputs=[{"expected_sequence": case["expected_sequence"]} for case in test_cases]
        )
        
        print(f"Created new dataset with {len(test_cases)} example(s)")
        return dataset
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None 