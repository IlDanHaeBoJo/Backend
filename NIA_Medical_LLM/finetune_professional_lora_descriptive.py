import os
import pandas as pd
import argparse
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from datasets import Dataset
from tqdm import tqdm
import json
from peft import get_peft_model, LoraConfig, TaskType
from utils import normalize_choices, normalize_numbering, load_full_data, generate_prompt_with_cot, generate_prompt_multiple, generate_prompt_descriptive


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune QWEN Model with LoRA on QA Data")
    parser.add_argument("--model_name_or_path", default="./results/qwen_14b_professional_lora_cot/checkpoint-56000", type=str)
    parser.add_argument("--output_dir", default="./results/qwen_14b_professional_lora_cot_v2", type=str)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--train_data_essential_path", default="./datasets/241030_[훈련용] 필수QA_ver01 (18627).csv", type=str)
    parser.add_argument("--train_data_professional_path", default="./datasets/241030_[훈련용] 전문QA_ver01 (14753).csv", type=str)
    parser.add_argument("--test_data_path", default="./datasets/241030_[테스트용] KorMedMCQA_ver01 (2494).csv", type=str)
    parser.add_argument("--per_device_train_batch_size", default=1, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=1, type=int)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--logging_steps", default=2000, type=int)
    parser.add_argument("--evaluation_strategy", default="steps", type=str)
    parser.add_argument("--eval_steps", default=2000, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--save_steps", default=4000, type=int)
    parser.add_argument("--save_total_limit", default=10, type=int)
    parser.add_argument("--report_to", default="wandb", type=str)
    parser.add_argument("--learning_rate", default=1e-6, type=float)  # Adjusted for LoRA
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument('--overwrite_output_dir', default=True, type=bool)
    return parser.parse_args()

def preprocess_data(data, tokenizer, is_test=False):
    def preprocess_fn(example):

        if example['q_type'] == 1:
            prompt = generate_prompt_multiple(example)
            print(f"Multiple: {prompt}")
        else:
            prompt = generate_prompt_descriptive(example)
            print(f"Descriptive: {prompt}")
        
        inputs = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=False,
            return_tensors="pt"
        )
        if not is_test:
            inputs["labels"] = inputs["input_ids"]
        return {k: v.squeeze().tolist() for k, v in inputs.items()}
    
    dataset = Dataset.from_pandas(data)
    return dataset.map(preprocess_fn, batched=False)

def main():
    args = parse_args()

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    #print(model)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Low-rank dimension
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]  # Specific to Qwen2ForCausalLM
    )
    model = get_peft_model(model, lora_config)
    
    # for name, param in model.named_parameters():
    #     print(f"{name}: requires_grad={param.requires_grad}")

    # Load and preprocess data
    train_data, test_data = load_full_data(args.train_data_professional_path, args.test_data_path)
    
    train_data['answer'] = train_data['answer'].apply(normalize_numbering)
    
    tokenized_train_dataset = preprocess_data(train_data, tokenizer)
    split_dataset = tokenized_train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print(train_dataset[0])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        learning_rate=args.learning_rate,
        gradient_checkpointing=False,
        remove_unused_columns=True,
        bf16=True,
        overwrite_output_dir=args.overwrite_output_dir,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model()

if __name__ == "__main__":
    main()
