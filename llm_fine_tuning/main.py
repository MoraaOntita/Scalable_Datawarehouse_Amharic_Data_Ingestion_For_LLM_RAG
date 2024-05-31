import os
from transformers import AutoTokenizer
from scripts.utils import load_config, setup_logging
from scripts.data_processing import DataProcessor
from scripts.model_training import ModelTrainer

def main():
    # Load configuration
    config_path = "./config/config.yaml"
    config = load_config(config_path)
    
    # Setup logging
    log_file = config['logging_directory'] + '/training.log'
    setup_logging(log_file)

    # Initialize tokenizer and data processor
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    data_processor = DataProcessor(model_name=config['model_name'], tokenizer=tokenizer)

    # Load and prepare datasets
    train_dataset = data_processor.prepare_dataset(config['data_directories']['train'])
    validation_dataset = data_processor.prepare_dataset(config['data_directories']['validation'])
    test_dataset = data_processor.prepare_dataset(config['data_directories']['test'])

    # Initialize model trainer
    model_trainer = ModelTrainer(model_name=config['model_name'], training_args=config['training_arguments'])

    # Train model
    model_trainer.train(train_dataset, validation_dataset)

    # Save model and tokenizer
    model_trainer.save_model(config['output_directory'], tokenizer)

if __name__ == "__main__":
    main()
