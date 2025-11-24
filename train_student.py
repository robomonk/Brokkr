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

# QLoRA Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# LoRA Config
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
        attn_implementation="flash_attention_2"
    )
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 3. Load Dataset
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_from_disk(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    # 4. Training Arguments (Standard HF Arguments)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Define formatting function
    def formatting_prompts_func(example):
        return example['text']

    # 5. Initialize Trainer
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,  # Passed here in trl 0.8.6
        packing=False,
    )

    # 6. Train
    print("Starting Training...")
    trainer.train()
    
    # 7. Save
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    print("Training Complete.")

if __name__ == "__main__":
    main()