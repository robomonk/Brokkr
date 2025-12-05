import torch
import argparse
import os
from datasets import load_from_disk
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="./processed_data/train")
    parser.add_argument("--eval_data_path", type=str, default="./processed_data/val")
    parser.add_argument("--output_dir", type=str, default="./student_txgemma_trace")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--model_id", type=str, default="google/txgemma-9b-chat")
    
    # TURBO DEFAULT
    parser.add_argument("--batch_size", type=int, default=4) # Adjusted for A100 40GB
    parser.add_argument("--grad_accum", type=int, default=2) # Matches original effective batch of 32
    parser.add_argument("--context_length", type=int, default=1024) # Reduced for Baseline
    parser.add_argument("--max_steps", type=int, default=-1) # Restore step limit control
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0:
        print(f"--- TURBO TRAINING (DDP Mode) ---")
        print(f"Model: {args.model_id}")
        print(f"Batch Size: {args.batch_size}")
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True
    )

    # 1. Load Model (DDP MODE)
    if local_rank == 0: print("Loading model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, attn_implementation="flash_attention_2", device_map={"": local_rank}
    )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    if local_rank == 0: model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = load_from_disk(args.train_data_path)
    eval_dataset = load_from_disk(args.eval_data_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.max_steps, # Allow limiting steps
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-4,
        #optim="paged_adamw_32bit",
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        bf16=True,
        max_grad_norm=0.3,
        report_to="tensorboard",
        logging_dir=args.log_dir,
        dataloader_num_workers=4, # <--- CRITICAL: Keep GPUs fed
        tf32=True, # <--- CRITICAL: A100 specific speedup
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # 5. Trainer with PACKING
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.context_length, 
        packing=True, # <--- THE SPEEDUP KEY (Concatenates samples)
    )

    if local_rank == 0: print("Starting Turbo Training...")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()