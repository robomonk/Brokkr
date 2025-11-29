import json
import os
import random
import sys
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Force output to terminal immediately
def log(msg):
    print(f"[INFO] {msg}", flush=True)

def log_error(msg):
    print(f"[ERROR] {msg}", flush=True)

# --- Configuration ---
INPUT_DATASET_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
OUTPUT_DIR = "./processed_data"
MAX_SAMPLES = 50000 
VAL_SPLIT_RATIO = 0.05

# Special Tokens
BACKTRACK_TOKEN = "<BACKTRACK>"
ERROR_TOKEN = "<ERROR_ANALYSIS>"

def get_tool_identifier(example):
    """Extracts the primary tool name to perform the split."""
    try:
        conversations = example.get('conversations', [])
        for msg in conversations:
            if msg['from'] == 'assistant':
                content = msg['value']
                # Look for Action string
                if "Action:" in content:
                    return content.split("Action:")[-1].split("\n")[0].strip()
                # Look for JSON tool call
                if "name" in content and "arguments" in content:
                    try:
                        return json.loads(content)['name']
                    except:
                        pass
        return "unknown_tool"
    except:
        return "unknown_tool"

def linearize_dfsdt_trace(example):
    """Transforms ShareGPT conversation into Linearized Failure-Aware format."""
    conversations = example.get('conversations', [])
    if not conversations: return None

    # Extract User Query
    query = next((msg['value'] for msg in conversations if msg['from'] == 'user'), None)
    if not query: return None

    # Llama-3 Format
    linearized_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n<think>\n"
    
    steps = [msg for msg in conversations if msg['from'] != 'user' and msg['from'] != 'system']
    if not steps: return None

    final_answer = steps[-1]['value']
    
    # Process steps
    for i, step in enumerate(steps[:-1]):
        content = step['value']
        role = step['from']
        
        if role == 'assistant':
            linearized_text += f"{content}\n"
        elif role == 'function' or role == 'tool':
            linearized_text += f"Observation: {content}\n"
            linearized_text += f"{ERROR_TOKEN} Analyzing previous result... {BACKTRACK_TOKEN}\n"

    linearized_text += f"Final Answer: {final_answer}</think><|eot_id|>"
    
    return {"text": linearized_text, "tool_id": get_tool_identifier(example)}

def main():
    log("Script starting...")
    
    # 1. Load Dataset
    log(f"Attempting to download/load dataset: {INPUT_DATASET_ID}...")
    try:
        # We explicitly verify the split exists
        raw_dataset = load_dataset(INPUT_DATASET_ID, split="train")
        log(f"Dataset loaded successfully. Total rows: {len(raw_dataset)}")
    except Exception as e:
        log_error(f"Failed to load dataset: {e}")
        return

    # 2. Linearization Loop
    log("Starting linearization process...")
    processed_samples = []
    tool_map = {} 
    
    # Adding tqdm for visual progress bar
    for sample in tqdm(raw_dataset, desc="Processing Traces", unit="sample"):
        if len(processed_samples) >= MAX_SAMPLES: 
            break
        try:
            linearized = linearize_dfsdt_trace(sample)
            if linearized:
                tid = linearized['tool_id']
                if tid not in tool_map: tool_map[tid] = []
                tool_map[tid].append(linearized)
                processed_samples.append(linearized)
        except Exception as e:
            # Silent skip for bad rows, but we continue
            continue

    log(f"Processing complete. Valid traces found: {len(processed_samples)}")
    log(f"Unique tools identified: {len(tool_map)}")

    if len(processed_samples) == 0:
        log_error("No samples were processed! Check dataset format.")
        return

    # 3. Splitting
    log("Performing Tool-Wise Split (Seen vs Unseen)...")
    unique_tools = list(tool_map.keys())
    random.seed(42)
    random.shuffle(unique_tools)
    
    split_idx = int(len(unique_tools) * (1 - VAL_SPLIT_RATIO))
    train_tools = set(unique_tools[:split_idx])
    val_tools = set(unique_tools[split_idx:])
    
    log(f"Total Tools: {len(unique_tools)} | Train Tools: {len(train_tools)} | Validation Tools: {len(val_tools)}")
    
    train_samples = []
    val_samples = []
    
    for sample in tqdm(processed_samples, desc="Splitting Data"):
        if sample['tool_id'] in train_tools:
            train_samples.append(sample)
        else:
            val_samples.append(sample)
            
    log(f"Final Count -> Train Samples: {len(train_samples)} | Validation Samples: {len(val_samples)}")

    # 4. Saving
    log(f"Saving data to {OUTPUT_DIR}...")
    try:
        os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/val", exist_ok=True)
        
        if train_samples:
            Dataset.from_list(train_samples).save_to_disk(f"{OUTPUT_DIR}/train")
            log("Train dataset saved.")
        else:
            log_error("Train dataset is empty!")

        if val_samples:
            Dataset.from_list(val_samples).save_to_disk(f"{OUTPUT_DIR}/val")
            log("Validation dataset saved.")
        else:
            log_error("Validation dataset is empty!")
            
    except Exception as e:
        log_error(f"Failed to save data: {e}")

    log("Script finished successfully.")

if __name__ == "__main__":
    main()