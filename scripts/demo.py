"""
Demonstration script for Taylor Expansion Learning.

This script demonstrates the end-to-end process of training and evaluating
different models for learning Taylor expansions.
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import sympy as sp

from taylor_expansion_learning.config import CONFIG
from taylor_expansion_learning.data.generator import TaylorDataGenerator
from taylor_expansion_learning.data.processor import DataProcessor
from taylor_expansion_learning.data.tokenizer import Tokenizer

from taylor_expansion_learning.models.lstm_model import LSTMSeq2Seq
from taylor_expansion_learning.models.improved_lstm_model import ImprovedLSTMSeq2Seq
from taylor_expansion_learning.models.transformer_model import TransformerSeq2Seq

from taylor_expansion_learning.training.lstm_trainer import LSTMTrainer
from taylor_expansion_learning.training.improved_lstm_trainer import ImprovedLSTMTrainer
from taylor_expansion_learning.training.transformer_trainer import TransformerTrainer

from taylor_expansion_learning.evaluation.evaluator import ModelEvaluator


def generate_dataset(config, save=True):
    """Generate and save the dataset."""
    print("\n=== Data Generation ===")
    
    data_generator = TaylorDataGenerator(config)
    functions, expansions = data_generator.generate_dataset()
    
    if save:
        dataset_path = os.path.join(config['paths']['data_dir'], 'taylor_dataset.json')
        data_generator.save_dataset(dataset_path)
    
    # Display a few examples
    print("\nSample data:")
    for i in range(5):
        print(f"\nFunction: {functions[i]}")
        print(f"Expansion: {expansions[i]}")
    
    return functions, expansions


def process_data(functions, expansions, config, device):
    """Process the data for model training."""
    print("\n=== Data Processing ===")
    
    data_processor = DataProcessor(config, device)
    
    # Ensure dataset size is divisible by batch_size to avoid issues
    batch_size = config['model']['batch_size']
    dataset_size = len(functions)
    usable_size = (dataset_size // batch_size) * batch_size
    
    print(f"Original dataset size: {dataset_size}")
    print(f"Usable size (multiple of batch_size {batch_size}): {usable_size}")
    
    # Use only the portion of the dataset that's divisible by batch_size
    functions = functions[:usable_size]
    expansions = expansions[:usable_size]
    
    return data_processor.process_data(functions, expansions)


def train_models(data, config, device, epochs=5):
    """Train all models and return the trainers."""
    print("\n=== Model Training ===")
    
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    tokenizer = data['tokenizer']
    vocab_size = data['vocab_size']
    
    # For demonstration, use fewer epochs
    demo_epochs = epochs
    print(f"Using {demo_epochs} epochs for demonstration")
    
    trainers = {}
    
    # 1. LSTM Model
    print("\nTraining LSTM Model...")
    lstm_model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout']
    ).to(device)
    
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=config['model']['learning_rate'])
    lstm_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN])
    
    lstm_trainer = LSTMTrainer(lstm_model, lstm_optimizer, lstm_criterion, tokenizer, config, device)
    lstm_trainer.train(train_loader, val_loader, demo_epochs)
    trainers['lstm'] = lstm_trainer
    
    # 2. Improved LSTM Model
    print("\nTraining Improved LSTM Model...")
    improved_lstm_model = ImprovedLSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=config['improved_lstm']['embedding_dim'],
        hidden_dim=config['improved_lstm']['hidden_dim'],
        num_layers=config['improved_lstm']['num_layers'],
        dropout=config['improved_lstm']['dropout'],
        attention_dim=config['improved_lstm']['attention_dim']
    ).to(device)
    
    improved_lstm_optimizer = optim.Adam(improved_lstm_model.parameters(), lr=config['model']['learning_rate'])
    improved_lstm_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN])
    
    improved_lstm_trainer = ImprovedLSTMTrainer(
        improved_lstm_model, improved_lstm_optimizer, improved_lstm_criterion, tokenizer, config, device
    )
    improved_lstm_trainer.train(train_loader, val_loader, demo_epochs)
    trainers['improved_lstm'] = improved_lstm_trainer
    
    # 3. Transformer Model
    print("\nTraining Transformer Model...")
    transformer_model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=config['transformer']['embedding_dim'],
        nhead=config['transformer']['num_heads'],
        num_encoder_layers=config['transformer']['num_encoder_layers'],
        num_decoder_layers=config['transformer']['num_decoder_layers'],
        dim_feedforward=config['transformer']['dim_feedforward'],
        dropout=config['transformer']['dropout']
    ).to(device)
    
    transformer_optimizer = optim.Adam(
        transformer_model.parameters(),
        lr=config['model']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    transformer_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN])
    
    transformer_trainer = TransformerTrainer(
        transformer_model, transformer_optimizer, transformer_criterion, tokenizer, config, device
    )
    transformer_trainer.train(train_loader, val_loader, demo_epochs)
    trainers['transformer'] = transformer_trainer
    
    return trainers


def compare_models(trainers, test_funcs, tokenizer, config):
    """Compare all trained models on test functions."""
    print("\n=== Model Comparison ===")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        trainers['lstm'], trainers['improved_lstm'], trainers['transformer'], tokenizer, config
    )
    
    # Use a subset of test functions for quick evaluation
    num_examples = min(30, len(test_funcs))
    print(f"Using {num_examples} examples for model comparison")
    test_funcs_subset = test_funcs[:num_examples]
    
    # Compare models
    results, metrics = evaluator.compare_models(test_funcs_subset)
    
    return results, metrics


def interactive_demo(trainers, tokenizer):
    """Interactive demo to generate Taylor expansions for user-input functions."""
    print("\n=== Interactive Demo ===")
    print("Enter mathematical functions to compute their Taylor expansions.")
    print("Type 'exit' to quit.")
    
    while True:
        function_text = input("\nEnter a function (e.g., 'sin(x)', 'exp(2*x)'): ")
        
        if function_text.lower() == 'exit':
            break
        
        try:
            # Compute true expansion using SymPy
            x = sp.Symbol('x')
            true_func = sp.sympify(function_text)
            true_expansion = sp.series(
                true_func,
                x,
                x0=CONFIG['data_generation']['x0'],
                n=CONFIG['data_generation']['expansion_order'] + 1
            ).removeO()
            
            print(f"\nTrue expansion: {true_expansion}")
            
            # Get predictions from each model
            start_time = time.time()
            lstm_pred = trainers['lstm'].predict(function_text)
            lstm_time = time.time() - start_time
            
            start_time = time.time()
            improved_lstm_pred = trainers['improved_lstm'].predict(function_text)
            improved_lstm_time = time.time() - start_time
            
            start_time = time.time()
            transformer_pred = trainers['transformer'].predict(function_text)
            transformer_time = time.time() - start_time
            
            # Print results with timing
            print(f"\nLSTM prediction ({lstm_time:.3f}s): {lstm_pred}")
            print(f"Improved LSTM prediction ({improved_lstm_time:.3f}s): {improved_lstm_pred}")
            print(f"Transformer prediction ({transformer_time:.3f}s): {transformer_pred}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please try another function.")


def main():
    """Main function to run the end-to-end demonstration."""
    print("=== Taylor Expansion Learning Demonstration ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if dataset exists
    dataset_path = os.path.join(CONFIG['paths']['data_dir'], 'taylor_dataset.json')
    
    if os.path.exists(dataset_path):
        print(f"Loading existing dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        functions = data['functions']
        expansions = data['expansions']
    else:
        print("Generating new dataset")
        functions, expansions = generate_dataset(CONFIG)
    
    # Process data
    processed_data = process_data(functions, expansions, CONFIG, device)
    
    # Train models
    trainers = train_models(processed_data, CONFIG, device, epochs=3)
    
    # Compare models
    test_funcs = processed_data['test_funcs']
    results, metrics = compare_models(trainers, test_funcs, processed_data['tokenizer'], CONFIG)
    
    # Interactive demo
    interactive_demo(trainers, processed_data['tokenizer'])
    
    print("\nDemonstration completed!")


if __name__ == "__main__":
    main()
