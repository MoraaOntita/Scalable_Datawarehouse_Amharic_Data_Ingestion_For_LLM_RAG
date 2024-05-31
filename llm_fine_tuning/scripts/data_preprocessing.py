import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Dict, Any

class DataProcessor:
    def __init__(self, model_name: str, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def load_data_from_directory(self, data_directory: str) -> pd.DataFrame:
        csv_files = [os.path.join(root, file)
                     for root, _, files in os.walk(data_directory)
                     for file in files if file.endswith(".csv")]

        data_frames = [pd.read_csv(file) for file in csv_files]
        combined_df = pd.concat(data_frames, ignore_index=True)
        return combined_df

    def prepare_dataset(self, data_directory: str) -> Dataset:
        combined_df = self.load_data_from_directory(data_directory)
        texts = combined_df['text'].tolist()
        tokenized_data = self.tokenizer(texts, truncation=True, padding=True)
        dataset = Dataset.from_dict({
            'input_ids': tokenized_data['input_ids'],
            'attention_mask': tokenized_data['attention_mask']
        })
        return dataset
