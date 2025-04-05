"""
Script to train the Improved LSTM model with attention for Taylor expansion learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from taylor_expansion_learning.config import CONFIG
from taylor_expansion_learning.data.generator import TaylorDataGenerator
from taylor_expansion_learning.data.processor import DataProcessor
from taylor_expansion_learning.models.improved_lstm_model import ImprovedLSTMSeq2Seq
from taylor_expansion_learning.training.improved_lstm_trainer import ImprovedLSTMTrainer


def main():
    """Train the Improved LSTM model with attention for Taylor expansion learning."""
    print("Starting Improved LSTM model training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset if it exists, otherwise generate it
    dataset_path = os.path.join(CONFIG['paths']['data_dir'], 'taylor_dataset.json')
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        import json
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        functions = data['functions']
        expansions = data['expansions']
    else:
        print("Generating new dataset")
        data_generator = TaylorDataGenerator(CONFIG)
        functions, expansions = data_generator.generate_dataset()
        data_generator.save_dataset(dataset_path)
    
    # Process data
    data_processor = DataProcessor(CONFIG, device)
    
    # Ensure dataset size is divisible by batch_size to avoid issues
    batch_size = CONFIG['model']['batch_size']
    dataset_size = len(functions)
    usable_size = (dataset_size // batch_size) * batch_size
    
    print(f"Original dataset size: {dataset_size}")
    print(f"Usable size (multiple of batch_size {batch_size}): {usable_size}")
    
    # Use only the portion of the dataset that's divisible by batch_size
    functions = functions[:usable_size]
    expansions = expansions[:usable_size]
    
    data = data_processor.process_data(functions, expansions)
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    tokenizer = data['tokenizer']
    vocab_size = data['vocab_size']
    
    # Initialize Improved LSTM model
    improved_lstm_model = ImprovedLSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['improved_lstm']['embedding_dim'],
        hidden_dim=CONFIG['improved_lstm']['hidden_dim'],
        num_layers=CONFIG['improved_lstm']['num_layers'],
        dropout=CONFIG['improved_lstm']['dropout'],
        attention_dim=CONFIG['improved_lstm']['attention_dim']
    ).to(device)
    
    # Initialize optimizer and criterion
    improved_lstm_optimizer = optim.Adam(improved_lstm_model.parameters(), 
                                      lr=CONFIG['model']['learning_rate'])
    improved_lstm_criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN]
    )
    
    # Initialize trainer
    improved_lstm_trainer = ImprovedLSTMTrainer(
        improved_lstm_model, improved_lstm_optimizer, improved_lstm_criterion, 
        tokenizer, CONFIG, device
    )
    
    # Train model
    improved_lstm_train_losses, improved_lstm_val_losses = improved_lstm_trainer.train(
        train_loader, val_loader, CONFIG['model']['epochs']
    )
    
    print("Improved LSTM model training completed!")


if __name__ == "__main__":
    main()
