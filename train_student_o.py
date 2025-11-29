import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
import os

# --- Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DATA_PATH = "./processed_data/train_linearized"
OUTPUT_DIR = "./student_8b_trace_distilled"
LOG_DIR = "./logs"

# QLoRA Config: 4-bit quantization for A100 40GB efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Config: Rank 64 to capture complex reasoning patterns
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)

def main():
    print(f"Loading model: {MODEL_ID}...")
    
    # 1. Load Base Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False, 
        attn_implementation="flash_attention_2" # Optimized for A100
    )
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load and Split Dataset
    print(f"Loading dataset from {DATA_PATH}...")
    full_dataset = load_from_disk(DATA_PATH)
    
    # Create a 5% hold-out set for Validation Loss monitoring
    dataset_dict = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
    
    print(f"Training Samples: {len(train_dataset)}")
    print(f"Validation Samples: {len(eval_dataset)}")

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,             # 1 Epoch is sufficient for SFT
        per_device_train_batch_size=4,  # Batch size 4 fits comfortably in 40GB
        gradient_accumulation_steps=4,  # Effective batch = 16
        learning_rate=2e-4,
        logging_steps=10,               # Log frequent updates
        save_strategy="steps",
        save_steps=50,                  # Save checkpoints often
        eval_strategy="steps",          
        eval_steps=50,
        load_best_model_at_end=True,    # Keep the best checkpoint
        fp16=False,
        bf16=True,                      # Use BFloat16 for A100
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="tensorboard",        
        logging_dir=LOG_DIR,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Define formatting function for the trainer
    def formatting_prompts_func(example):
        # Returns the text column which contains the linearized trace
        return example['text']

    # 5. Initialize Trainer
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,            # Truncate very long traces
        packing=False,
    )

    # 6. Train
    print("Starting Training...")
    trainer.train()
    
    # 7. Save Final Model
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    print("Training Complete.")

if __name__ == "__main__":
    main()