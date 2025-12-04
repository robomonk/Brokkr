import torch
import argparse
import json
import ast
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
import numpy as np

# --- Configuration ---
MODEL_ID = "google/txgemma-9b-chat"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./processed_data/val_biotools")
    return parser.parse_args()

# --- Biomni Scoring Logic (Ported from biomni_eval1.py) ---
def compute_biomni_reward(task_name, user_answer, ground_truth):
    """Strict functional correctness check from Biomni paper"""
    try:
        # Normalize
        u_ans = user_answer.strip()
        g_ans = str(ground_truth).strip()

        # 1. Exact Match (CRISPR, GWAS Variant)
        if task_name in ["gwas_variant_prioritization", "crispr_delivery"]:
            return 1.0 if u_ans.lower() == g_ans.lower() else 0.0
            
        # 2. JSON / Dict Parsing (Rare Disease, Patient Gene)
        if task_name in ["rare_disease_diagnosis", "drug_repurposing"]:
            try:
                # Attempt to extract JSON from the text trace
                # We look for the last JSON-like object
                if "{" in u_ans:
                    u_ans = u_ans[u_ans.find("{") : u_ans.rfind("}")+1]
                
                u_dict = json.loads(u_ans) if isinstance(u_ans, str) else u_ans
                
                # Heuristic: Biomni ground truth is sometimes a string representation of a dict
                if isinstance(g_ans, str) and "{" in g_ans:
                    g_dict = ast.literal_eval(g_ans)
                else:
                    g_dict = g_ans

                # Compare Critical Keys (e.g., OMIM_ID or Drug ID)
                # We allow fuzzy matching on keys for robustness
                for k in ["OMIM_ID", "disease_name", "drug_id"]:
                    if k in g_dict:
                        return 1.0 if str(u_dict.get(k)) == str(g_dict[k]) else 0.0
            except:
                return 0.0
                
        return 0.0 # Default fail
    except:
        return 0.0

def main():
    args = parse_args()
    
    print(f"Evaluating {args.adapter_path} on Biomni Benchmark...")
    
    # Load Model
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    dataset = load_from_disk(args.data_path)
    
    scores = []
    corrections = 0
    backtracks = 0
    
    print(f"\n--- Evaluation Loop ({len(dataset)} samples) ---")
    
    for i, example in enumerate(dataset):
        # Generate
        input_text = example['text']
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        
        full_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the "Final Answer" part (or the last tool call)
        # For our purpose, we treat the text after "Final Answer:" as the answer
        # OR the arguments of the last Action if no final answer.
        generated_answer = ""
        if "Final Answer:" in full_out:
            generated_answer = full_out.split("Final Answer:")[-1].strip()
        elif "Action:" in full_out:
            generated_answer = full_out.split("Action:")[-1].split("\n")[0].strip()
            
        # Calculate Score
        score = compute_biomni_reward(example['task_type'], generated_answer, example['ground_truth'])
        scores.append(score)
        
        # Track Meta-Cognition
        if "<BACKTRACK>" in full_out:
            backtracks += 1
            if score == 1.0: corrections += 1
            
        if i % 10 == 0:
            print(f"[{i}] Task: {example['task_type']} | Score: {score} | Backtrack: {1 if '<BACKTRACK>' in full_out else 0}")

    print("\n" + "="*40)
    print(f"Biomni Pass Rate:  {np.mean(scores)*100:.2f}%")
    print(f"Correction Rate:   {(corrections/backtracks*100) if backtracks else 0:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()