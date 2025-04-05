"""
Script to train the Transformer model for Taylor expansion learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from taylor_expansion_learning.config import CONFIG
from taylor_expansion_learning.data.generator import TaylorDataGenerator
from taylor_expansion_learning.data.processor import DataProcessor
from taylor_expansion_learning.models.transformer_model import TransformerSeq2Seq
from taylor_expansion_learning.training.transformer_trainer import TransformerTrainer


def main():
    """Train the Transformer model for Taylor expansion learning."""
    print("Starting Transformer model training")
    
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
    
    # Initialize Transformer model
    transformer_model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=CONFIG['transformer']['embedding_dim'],
        nhead=CONFIG['transformer']['num_heads'],
        num_encoder_layers=CONFIG['transformer']['num_encoder_layers'],
        num_decoder_layers=CONFIG['transformer']['num_decoder_layers'],
        dim_feedforward=CONFIG['transformer']['dim_feedforward'],
        dropout=CONFIG['transformer']['dropout']
    ).to(device)
    
    # Initialize optimizer with beta parameters for better training stability
    transformer_optimizer = optim.Adam(
        transformer_model.parameters(),
        lr=CONFIG['model']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    transformer_criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN]
    )
    
    # Initialize trainer
    transformer_trainer = TransformerTrainer(
        transformer_model, transformer_optimizer, transformer_criterion, 
        tokenizer, CONFIG, device
    )
    
    # Train model
    transformer_train_losses, transformer_val_losses = transformer_trainer.train(
        train_loader, val_loader, CONFIG['model']['epochs']
    )
    
    print("Transformer model training completed!")


if __name__ == "__main__":
    main()
