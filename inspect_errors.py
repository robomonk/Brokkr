import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
import pandas as pd

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./student_8b_trace_distilled"  # Path to your saved QLoRA adapter
DATA_PATH = "./processed_data/train_linearized"

def calculate_per_sample_loss(model, tokenizer, dataset):
    """
    Iterates through the dataset and computes the Cross-Entropy loss for each sample.
    """
    results = []
    
    print("Computing loss for each sample...")
    
    # We use batch_size=1 for precise per-sample loss attribution
    # In a real scenario, you could batch this, but strict per-sample analysis is safer here.
    for i, sample in tqdm(enumerate(dataset), total=len(dataset)):
        text = sample['text']
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
        
        # Forward pass (no gradient needed)
        with torch.no_grad():
            # labels=input_ids automatically triggers CrossEntropyLoss inside the model
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            
        results.append({
            "index": i,
            "loss": loss,
            "text": text
        })
        
    return pd.DataFrame(results)

def main():
    # 1. Load 4-bit Config (Must match training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 2. Load Base Model
    print("Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    # 3. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Load Trained Adapter
    print(f"Loading Adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() # Set to evaluation mode

    # 5. Load Validation Dataset
    # We look at the 'test' split we created during training splitting
    print(f"Loading dataset from {DATA_PATH}...")
    full_dataset = load_from_disk(DATA_PATH)
    # Re-create the same split seed to ensure we look at the validation set
    eval_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)['test']
    
    # 6. Compute Losses
    df = calculate_per_sample_loss(model, tokenizer, eval_dataset)
    
    # 7. Sort and Extract
    df_sorted = df.sort_values(by="loss", ascending=False) # Highest loss first
    
    highest_error = df_sorted.head(10)
    lowest_error = df_sorted.tail(10)

    # 8. Display Results
    print("\n" + "="*50)
    print("TOP 10 HIGHEST ERROR SAMPLES (Hardest to Learn)")
    print("="*50)
    for idx, row in highest_error.iterrows():
        print(f"\n[Rank {idx+1}] Loss: {row['loss']:.4f}")
        print(f"Snippet: {row['text'][:300]}...") # Print first 300 chars
        print("-" * 20)

    print("\n" + "="*50)
    print("TOP 10 LOWEST ERROR SAMPLES (Easiest to Learn)")
    print("="*50)
    for idx, row in lowest_error.iterrows():
        print(f"\n[Rank {idx+1}] Loss: {row['loss']:.4f}")
        print(f"Snippet: {row['text'][:300]}...")
        print("-" * 20)

    # Optional: Save to CSV for deep analysis
    df.to_csv("error_analysis_report.csv", index=False)
    print("\nFull error report saved to 'error_analysis_report.csv'")

if __name__ == "__main__":
    main()