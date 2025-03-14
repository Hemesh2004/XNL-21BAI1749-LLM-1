import os
import deepspeed
import torch
from transformers import (
    GPTJForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb
from dotenv import load_dotenv
from data_processing import DataProcessor
from monitoring import MonitoringCallback, ModelEvaluator
from security import SecurityManager

# Load environment variables
load_dotenv()

def setup_model_and_tokenizer():
    """Initialize GPT-J model and tokenizer with DeepSpeed configuration"""
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    return model, tokenizer

def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="gpt-j-finetuning")
    
    # Initialize security manager
    security_manager = SecurityManager()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Initialize data processor
    data_processor = DataProcessor(tokenizer_name="EleutherAI/gpt-j-6B")
    
    # Prepare dataset with preprocessing and augmentation
    processed_datasets = data_processor.load_and_preprocess(
        dataset_path=os.getenv("DATASET_PATH", "your_dataset"),
        text_column="text",
        max_length=512,
        train_size=0.8
    )
    
    # Initialize monitoring callback
    monitoring_callback = MonitoringCallback(log_dir="./logs")
    
    # Initialize model evaluator
    model_evaluator = ModelEvaluator(model, tokenizer)
    
    # Initialize DeepSpeed configuration
    ds_config = "deepspeed_config.json"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=32,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        evaluation_strategy="steps",
        deepspeed=ds_config,
        fp16=True,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        distributed_state=deepspeed.init_distributed(),
    )
    
    # Initialize trainer with monitoring callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[monitoring_callback]
    )
    
    # Start training
    trainer.train()
    
    # Final evaluation
    eval_results = []
    test_prompts = [
        "This is a test prompt",
        "Another test prompt",
        "Final test prompt"
    ]
    eval_results = model_evaluator.evaluate_batch(test_prompts)
    
    # Log evaluation results
    wandb.log({"final_evaluation": eval_results})
    
    # Save final model with watermark and encryption
    model_path = "./final_model"
    trainer.save_model(model_path)
    
    # Add watermark and encrypt model
    model_state = model.state_dict()
    watermarked_state = security_manager.watermark_model(model_state)
    model.load_state_dict(watermarked_state)
    torch.save(model.state_dict(), f"{model_path}/pytorch_model.bin")
    security_manager.encrypt_model(f"{model_path}/pytorch_model.bin")

if __name__ == "__main__":
    main()
