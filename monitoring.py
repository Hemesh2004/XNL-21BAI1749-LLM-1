processed_datasets = data_processor.load_and_preprocess(
    dataset_path="your_dataset",  # Update this
    text_column="text",
    max_length=512,
    train_size=0.8
)import wandb

import torch
from typing import Dict, Any, List
import numpy as np
from dataclasses import dataclass
from transformers import TrainerCallback
import logging
import json
from pathlib import Path

@dataclass
class TrainingMetrics:
    loss: float
    perplexity: float
    gpu_memory_used: float
    learning_rate: float

class MonitoringCallback(TrainerCallback):
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            filename=self.log_dir / "training.log",
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize WandB
        wandb.init(project="gpt-j-finetuning", config={
            "model": "GPT-J-6B",
            "optimizer": "AdamW",
            "scheduler": "linear_warmup"
        })
        
    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs):
        """Log metrics after each step"""
        if state.global_step % args.logging_steps == 0:
            metrics = self._gather_metrics(state, kwargs.get("metrics", {}))
            self._log_metrics(metrics, state.global_step)
            
    def _gather_metrics(self, state: Any, metrics: Dict) -> TrainingMetrics:
        """Gather training metrics"""
        # Get GPU memory usage
        gpu_memory = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        
        return TrainingMetrics(
            loss=metrics.get("loss", 0.0),
            perplexity=torch.exp(torch.tensor(metrics.get("loss", 0.0))).item(),
            gpu_memory_used=gpu_memory,
            learning_rate=state.learning_rate
        )
    
    def _log_metrics(self, metrics: TrainingMetrics, step: int):
        """Log metrics to WandB and local file"""
        # Log to WandB
        wandb.log({
            "loss": metrics.loss,
            "perplexity": metrics.perplexity,
            "gpu_memory_gb": metrics.gpu_memory_used,
            "learning_rate": metrics.learning_rate
        }, step=step)
        
        # Log to file
        log_entry = {
            "step": step,
            "metrics": metrics.__dict__,
            "timestamp": wandb.env.get_log_timestamp()
        }
        
        logging.info(f"Step {step}: {json.dumps(log_entry)}")
        
    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Dict[str, float], **kwargs):
        """Log evaluation metrics"""
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        wandb.log(eval_metrics, step=state.global_step)
        logging.info(f"Evaluation metrics: {json.dumps(eval_metrics)}")
        
    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs):
        """Cleanup and final logging"""
        # Save final model metrics
        final_metrics = {
            "final_loss": state.log_history[-1].get("loss", 0.0),
            "total_steps": state.global_step,
            "training_time": state.training_time
        }
        
        # Log final metrics
        wandb.log(final_metrics)
        logging.info(f"Training completed. Final metrics: {json.dumps(final_metrics)}")
        
        # Close WandB
        wandb.finish()

class ModelEvaluator:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_sample(self, prompt: str, max_length: int = 100) -> Dict[str, Any]:
        """Generate and evaluate a single sample"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "length": len(outputs[0])
        }
    
    def evaluate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Evaluate a batch of prompts"""
        return [self.evaluate_sample(prompt) for prompt in prompts]
