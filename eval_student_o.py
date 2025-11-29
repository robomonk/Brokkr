import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import os

# --- Configuration ---
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_PATH = "./student_8b_trace_distilled" # Path to your trained weights
DATASET_ID = "Yhyu13/ToolBench_toolllama_G123_dfs"
LOG_DIR = "./logs/evaluation_traces"

def main():
    # Initialize TensorBoard Writer
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    print("Loading Base Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    print(f"Loading Trained Adapter from {ADAPTER_PATH}...")
    # Load the fine-tuned LoRA adapters
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    print("Loading Test Data (Unseen)...")
    # We use the 'test' split from the original dataset. 
    # These are examples the model did NOT see during training.
    test_dataset = load_dataset(DATASET_ID, split="test[:10]") 
    
    print("\n=== STARTING EVALUATION ===\n")
    
    for i, example in enumerate(test_dataset):
        # Extract query from conversation history
        # The dataset format is a list of messages; we find the first user message.
        conversation = example['conversations']
        query = next((msg['value'] for msg in conversation if msg['from'] == 'user'), None)
        
        if not query: continue
        
        # Format Input using Llama-3 Template
        # We prompt the model up to the start of the assistant's response
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        print(f"--- Example {i+1} ---")
        print(f"Query: {query}\n")
        
        # Generate Reasoning Trace
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1, # Low temp for deterministic, logical reasoning
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the newly generated trace (after the prompt)
        generated_trace = response.split("assistant")[-1].strip()
        
        print(f"Generated Trace:\n{generated_trace}")
        print("="*60 + "\n")
        
        # --- LOGGING TO TENSORBOARD ---
        # This creates a card in the "Text" tab of TensorBoard
        log_text = f"**Query:** {query}  \n\n**Generated Trace:** \n{generated_trace}"
        writer.add_text("Zero-Shot_Generalization", log_text, global_step=i)

    writer.close()
    print(f"Evaluation complete. Run 'tensorboard --logdir ./logs' to view traces.")

if __name__ == "__main__":
    main()