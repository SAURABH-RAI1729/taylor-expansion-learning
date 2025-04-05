"""
Configuration settings for the Taylor Expansion Learning project.

This module contains global configuration parameters for data generation,
tokenization, model architectures, and training hyperparameters.
"""

import os

# Configuration dictionary with all parameters for the project
CONFIG = {
    'data_generation': {
        'num_functions': 10000,    # Number of functions to generate
        'expansion_order': 4,     # Order of Taylor expansion
        'x0': 0,                  # Point around which to expand
    },
    'tokenization': {
        'max_function_length': 50,   # Maximum length for function tokens
        'max_expansion_length': 100, # Maximum length for expansion tokens
    },
    'model': {
        'embedding_dim': 128,     # Dimension of token embeddings
        'hidden_dim': 256,        # Hidden dimension for LSTM
        'num_layers': 2,          # Number of LSTM layers
        'dropout': 0.2,           # Dropout rate
        'batch_size': 32,         # Batch size for training
        'learning_rate': 0.001,   # Learning rate
        'epochs': 30,             # Number of training epochs
    },
    'improved_lstm': {
        'embedding_dim': 256,     # Larger embedding dimension
        'hidden_dim': 512,        # Larger hidden dimension
        'num_layers': 3,          # More LSTM layers
        'dropout': 0.3,           # Slightly higher dropout
        'attention_dim': 256,     # Attention mechanism dimension
    },
    'transformer': {
        'embedding_dim': 128,        # Dimension of token embeddings
        'num_heads': 8,              # Number of attention heads
        'num_encoder_layers': 4,     # Number of encoder layers
        'num_decoder_layers': 4,     # Number of decoder layers
        'dim_feedforward': 1024,      # Dimension of feedforward layer
        'dropout': 0.2,              # Dropout rate
    },
    'paths': {
        'data_dir': 'data',           # Directory for storing generated data
        'models_dir': 'models',       # Directory for storing trained models
        'results_dir': 'results',     # Directory for storing evaluation results
    }
}

# Create directories if they don't exist
for dir_path in CONFIG['paths'].values():
    os.makedirs(dir_path, exist_ok=True)
