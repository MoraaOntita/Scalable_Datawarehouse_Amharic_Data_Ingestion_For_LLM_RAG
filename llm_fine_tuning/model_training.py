import logging
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import DatasetDict
from typing import Dict, Any
import os

class ModelTrainer:
    def __init__(self, model_name: str, training_args: Dict[str, Any]):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.training_args = TrainingArguments(**training_args)

    def train(self, train_dataset: Dataset, eval_dataset: Dataset) -> None:
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

    def save_model(self, save_directory: str, tokenizer) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        logging.info(f"Model and tokenizer saved to {save_directory}")
