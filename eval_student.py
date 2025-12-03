import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="google/txgemma-9b-chat")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Evaluating Adapter: {args.adapter_path}")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config, device_map="auto", attn_implementation="flash_attention_2")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    val_dataset = load_from_disk(args.data_path)
    if args.max_samples > 0: val_dataset = val_dataset.select(range(args.max_samples))

    total = 0; solved = 0; backtracks = 0; corrections = 0
    
    print("\n--- RESULTS ---")
    for i, example in enumerate(val_dataset):
        full_text = example['text']
        prompt_end = full_text.find("<start_of_turn>model") + len("<start_of_turn>model\n")
        prompt = full_text[:prompt_end]
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        is_solved = "Final Answer:" in generated
        has_backtrack = "<BACKTRACK>" in generated or "Analyzing previous result" in generated
        
        total += 1
        if is_solved: solved += 1
        if has_backtrack:
            backtracks += 1
            if is_solved: corrections += 1
            
        print(f"Sample {i}: Backtrack={has_backtrack}, Solved={is_solved}")

    print("\n" + "="*30)
    print(f"Pass Rate:       {solved/total*100:.2f}%")
    print(f"Correction Rate: {(corrections/backtracks*100) if backtracks else 0:.2f}%")
    print("="*30)

if __name__ == "__main__":
    main()