"""
Script to train the LSTM model for Taylor expansion learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from taylor_expansion_learning.config import CONFIG
from taylor_expansion_learning.data.generator import TaylorDataGenerator
from taylor_expansion_learning.data.processor import DataProcessor
from taylor_expansion_learning.models.lstm_model import LSTMSeq2Seq
from taylor_expansion_learning.training.lstm_trainer import LSTMTrainer


def main():
    """Train the LSTM model for Taylor expansion learning."""
    print("Starting LSTM model training")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    data_generator = TaylorDataGenerator(CONFIG)
    functions, expansions = data_generator.generate_dataset()
    
    # Save raw dataset
    dataset_path = os.path.join(CONFIG['paths']['data_dir'], 'taylor_dataset.json')
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
    
    # Initialize LSTM model
    lstm_model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['model']['embedding_dim'],
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers'],
        dropout=CONFIG['model']['dropout']
    ).to(device)
    
    # Initialize optimizer and criterion
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=CONFIG['model']['learning_rate'])
    lstm_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN])
    
    # Initialize trainer
    lstm_trainer = LSTMTrainer(lstm_model, lstm_optimizer, lstm_criterion, tokenizer, CONFIG, device)
    
    # Train model
    lstm_train_losses, lstm_val_losses = lstm_trainer.train(
        train_loader, val_loader, CONFIG['model']['epochs']
    )
    
    print("LSTM model training completed!")


if __name__ == "__main__":
    main()

