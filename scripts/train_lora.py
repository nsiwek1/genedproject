#!/usr/bin/env python3
"""
LoRA fine-tuning script for Confucius/Mencius datasets.

Usage (local with 1B model):
    python3 scripts/train_lora.py \
        --base-model meta-llama/Llama-3.2-1B-Instruct \
        --confucius-data data/instruction_pairs/confucius_auto.jsonl \
        --mencius-data data/instruction_pairs/mencius_auto.jsonl \
        --output models/confucius-mencius-lora-1b \
        --batch-size 2 \
        --grad-accum 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from huggingface_hub import login, HfFolder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Llama with LoRA for Confucius/Mencius personas.")
    parser.add_argument("--base-model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Base causal LM checkpoint (default: 1B for local training).")
    parser.add_argument("--confucius-data", type=Path, default=Path("data/instruction_pairs/confucius_auto.jsonl"), help="JSONL file with Confucius instruction pairs.")
    parser.add_argument("--mencius-data", type=Path, default=Path("data/instruction_pairs/mencius_auto.jsonl"), help="JSONL file with Mencius instruction pairs.")
    parser.add_argument("--output", type=Path, default=Path("models/confucius-mencius-lora-1b"), help="Directory to save adapter weights.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size (default: 2 for local 1B training).")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16 for local training).")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization (use full precision).")
    parser.add_argument("--cpu-only", action="store_true", help="Force CPU training (slower, but works without GPU).")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face token for accessing gated models. Can also set HF_TOKEN env var.")
    return parser.parse_args()


def load_and_prepare_dataset(paths: List[Path]) -> Dataset:
    data_files = [str(p) for p in paths]
    ds = load_dataset("json", data_files=data_files, split="train")

    def format_row(row):
        philosopher = row["philosopher"]
        tag = f"[{philosopher}] "
        user = row["messages"][1]["content"]
        assistant = row["messages"][2]["content"]
        return {"text": tag + user + "\n\n" + assistant}

    return ds.map(format_row)


def main():
    import torch
    
    args = parse_args()
    
    # Handle Hugging Face authentication
    hf_token = args.hf_token or os.getenv("HF_TOKEN") or HfFolder.get_token()
    if hf_token:
        print("Authenticating with Hugging Face...")
        login(token=hf_token)
        print("Authentication successful.")
    else:
        print("\n⚠️  WARNING: No Hugging Face token found!")
        print("   Llama models require authentication. Please:")
        print("   1. Get a token from https://huggingface.co/settings/tokens")
        print("   2. Request access to meta-llama/Llama-3.2-1B-Instruct at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("   3. Run: export HF_TOKEN=your_token_here")
        print("   4. Or pass: --hf-token your_token_here")
        print("\n   Attempting to continue anyway (may fail if model is gated)...\n")
    
    print(f"Starting training with base model: {args.base_model}")
    print(f"Output directory: {args.output}")
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.grad_accum}")
    print(f"Epochs: {args.epochs}, Learning rate: {args.lr}")
    
    if args.cpu_only:
        print("CPU-only mode enabled (training will be slower).")
    else:
        cuda_available = torch.cuda.is_available()
        print(f"GPU available: {cuda_available}")
        if cuda_available:
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("Note: Running on CPU. 8-bit quantization will be disabled (requires CUDA).")

    print("\nLoading dataset...")
    dataset = load_and_prepare_dataset([args.confucius_data, args.mencius_data])
    print(f"Dataset size: {len(dataset)} examples")

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading model...")
    model_kwargs = {}
    
    # Only use 8-bit quantization if CUDA is available and not disabled
    use_8bit = not args.cpu_only and not args.no_8bit and torch.cuda.is_available()
    
    if use_8bit:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
            print("Using 8-bit quantization with auto device mapping.")
        except Exception as e:
            print(f"Warning: 8-bit loading failed ({e}), falling back to full precision.")
            model_kwargs = {}
    else:
        if args.cpu_only:
            print("CPU-only mode: Using full precision model loading.")
        elif args.no_8bit:
            print("8-bit quantization disabled: Using full precision.")
        elif not torch.cuda.is_available():
            print("No CUDA available: Using full precision model loading (CPU mode).")
        else:
            print("Using full precision model loading.")
    
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=50,
        save_steps=500,
        fp16=not args.cpu_only and torch.cuda.is_available(),
        no_cuda=args.cpu_only,
    )
    
    print(f"\nTraining configuration:")
    print(f"  Output: {args.output}")
    print(f"  FP16: {training_args.fp16}")
    print(f"  Device: {'CPU' if args.cpu_only else 'GPU' if torch.cuda.is_available() else 'CPU (fallback)'}")
    print("\nStarting training...\n")

    # SFTTrainer API: use formatting_func to extract text from dataset
    # The function receives a dict (one example) and returns a string
    def formatting_func(example):
        # Extract the "text" field from the example
        return example["text"]
    
    # SFTTrainer in this version uses formatting_func instead of dataset_text_field
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
        max_seq_length=args.max_seq_len,
        processing_class=tokenizer,  # Pass tokenizer as processing_class
    )

    trainer.train()
    
    print("\nTraining complete! Saving model and tokenizer...")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"\nModel saved to: {args.output}")
    print("You can now use this directory with the Gradio chatbot interface.")


if __name__ == "__main__":
    main()

