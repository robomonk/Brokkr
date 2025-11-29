import json
from datasets import load_dataset
import os

# Configuration
DATASET_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
OUTPUT_DIR = "./processed_data"

def normalize_message(msg):
    """
    Adapts diverse dataset formats to a standard {'role': ..., 'content': ...}
    """
    # Map 'from' -> 'role' (ShareGPT format)
    if 'from' in msg:
        role_map = {
            'human': 'user',
            'gpt': 'assistant',
            'function': 'tool',
            'system': 'system'
        }
        return {
            'role': role_map.get(msg['from'], msg['from']),
            'content': msg.get('value', '')
        }
    
    # Standard format
    return msg

def linearize_trace(example):
    """
    Transforms a ToolBench DFSDT trace into the Llama-3 format.
    Handles both 'role/content' and 'from/value' schemas.
    """
    # 1. Get the conversation list
    if 'conversations' in example:
        raw_conversation = example['conversations']
    elif 'messages' in example:
        raw_conversation = example['messages']
    else:
        return {"text": ""}

    if not raw_conversation:
        return {"text": ""}

    # 2. Normalize the conversation structure
    conversation = [normalize_message(m) for m in raw_conversation]

    # 3. Extract User Query
    query = ""
    for msg in conversation:
        if msg['role'] == 'user':
            query = msg['content']
            break
            
    if not query: 
        return {"text": ""}

    # 4. Build Linearized String (Llama-3 Format)
    # <|begin_of_text|><|start_header_id|>user...
    linearized_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    # 5. Iterate through turns
    for msg in conversation:
        role = msg['role']
        content = msg['content']
        
        # Skip the initial user query (already added)
        if role == 'user' and content == query:
            continue
            
        if role == 'tool':
            # Observation
            linearized_text += f"<|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|>"
        elif role == 'assistant':
            # Thought + Action (The Trace)
            linearized_text += f"{content}"
            
    linearized_text += "<|eot_id|>"
    
    return {"text": linearized_text}

def main():
    print(f"Downloading dataset: {DATASET_ID}...")
    try:
        # Load 5% for a robust test (about 9,000 examples)
        # If this works, you can change it to split="train" for the full run
        dataset = load_dataset(DATASET_ID, split="train[:5%]") 
        
        print(f"Sample loaded. Size: {len(dataset)}")
        # Debug: print keys of first message to confirm fix
        first_msg = dataset[0]['conversations'][0]
        print(f"Detected Schema Keys: {list(first_msg.keys())}")
        
        print("Linearizing traces...")
        processed_dataset = dataset.map(linearize_trace, remove_columns=dataset.column_names)
        
        # Filter empty rows
        processed_dataset = processed_dataset.filter(lambda x: len(x['text']) > 20)
        
        print("\n--- Sample Linearized Trace (First 500 chars) ---")
        print(processed_dataset[0]['text'][:500])
        print("-------------------------------------------------\n")
        
        # Save
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "train_linearized")
        processed_dataset.save_to_disk(save_path)
        print(f"Success! Processed data saved to: {save_path}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()