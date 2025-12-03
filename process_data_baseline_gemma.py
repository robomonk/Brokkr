import json
import os
import random
from datasets import load_dataset, Dataset
from tqdm import tqdm

INPUT_DATASET_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
OUTPUT_DIR = "./processed_data_baseline"
MAX_SAMPLES = None
VAL_SPLIT_RATIO = 0.05 

def get_tool_identifier(example):
    try:
        conversations = example.get('conversations', [])
        for msg in conversations:
            if msg['from'] == 'assistant' and "Action:" in msg['value']:
                return msg['value'].split("Action:")[-1].split("\n")[0].strip()
        return "unknown_tool"
    except:
        return "unknown_tool"

def linearize_standard_sft(example):
    conversations = example.get('conversations', [])
    if not conversations: return None

    query = next((msg['value'] for msg in conversations if msg['from'] == 'user'), None)
    if not query: return None

    linearized_text = f"<start_of_turn>user\n{query}<end_of_turn>\n<start_of_turn>model\n<think>\n"
    
    steps = [msg for msg in conversations if msg['from'] != 'user' and msg['from'] != 'system']
    final_answer = ""
    if steps:
        final_answer = steps[-1]['value']
        for i, step in enumerate(steps[:-1]):
            content = step['value']
            role = step['from']
            # Filter out failures
            if "failed" in content.lower() or "error" in content.lower(): continue

            if role == 'assistant': linearized_text += f"{content}\n"
            elif role == 'function' or role == 'tool': linearized_text += f"Observation: {content}\n"

    linearized_text += f"Final Answer: {final_answer}</think><end_of_turn>"
    return {"text": linearized_text, "tool_id": get_tool_identifier(example)}

def main():
    print(f"Loading {INPUT_DATASET_ID}...")
    raw_dataset = load_dataset(INPUT_DATASET_ID, split="train")

    print("Linearizing traces (Baseline)...")
    processed_samples = []
    tool_map = {} 
    
    for sample in tqdm(raw_dataset):
        if MAX_SAMPLES and len(processed_samples) >= MAX_SAMPLES: break
        try:
            linearized = linearize_standard_sft(sample)
            if linearized:
                tid = linearized['tool_id']
                if tid not in tool_map: tool_map[tid] = []
                tool_map[tid].append(linearized)
                processed_samples.append(linearized)
        except: continue

    unique_tools = list(tool_map.keys())
    random.seed(42)
    random.shuffle(unique_tools)
    split_idx = int(len(unique_tools) * (1 - VAL_SPLIT_RATIO))
    train_tools = set(unique_tools[:split_idx])
    
    train_samples = []
    val_samples = []
    for sample in processed_samples:
        if sample['tool_id'] in train_tools: train_samples.append(sample)
        else: val_samples.append(sample)
            
    os.makedirs(f"{OUTPUT_DIR}/train", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/val", exist_ok=True)
    Dataset.from_list(train_samples).save_to_disk(f"{OUTPUT_DIR}/train")
    Dataset.from_list(val_samples).save_to_disk(f"{OUTPUT_DIR}/val")
    print(f"Saved Baseline Data: {len(train_samples)} Train | {len(val_samples)} Val")

if __name__ == "__main__":
    main()