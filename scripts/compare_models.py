"""
Script to compare all models for Taylor expansion learning.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from taylor_expansion_learning.config import CONFIG
from taylor_expansion_learning.data.processor import DataProcessor
from taylor_expansion_learning.data.tokenizer import Tokenizer

from taylor_expansion_learning.models.lstm_model import LSTMSeq2Seq
from taylor_expansion_learning.models.improved_lstm_model import ImprovedLSTMSeq2Seq
from taylor_expansion_learning.models.transformer_model import TransformerSeq2Seq

from taylor_expansion_learning.training.lstm_trainer import LSTMTrainer
from taylor_expansion_learning.training.improved_lstm_trainer import ImprovedLSTMTrainer
from taylor_expansion_learning.training.transformer_trainer import TransformerTrainer

from taylor_expansion_learning.evaluation.evaluator import ModelEvaluator


def main():
    """Compare all models for Taylor expansion learning."""
    print("Starting model comparison")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset_path = os.path.join(CONFIG['paths']['data_dir'], 'taylor_dataset.json')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Please run one of the training scripts first.")
        return
    
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    functions = data['functions']
    expansions = data['expansions']
    
    # Load tokenizer
    tokenizer_path = os.path.join(CONFIG['paths']['data_dir'], 'tokenizer.json')
    
    if os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.load(tokenizer_path)
    else:
        print("Tokenizer not found. Creating a new one.")
        tokenizer = Tokenizer()
        tokenizer.build_vocab(functions + expansions)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Check if models exist
    models_dir = CONFIG['paths']['models_dir']
    lstm_model_path = os.path.join(models_dir, 'best_lstm_model.pt')
    improved_lstm_model_path = os.path.join(models_dir, 'best_improved_lstm_model.pt')
    transformer_model_path = os.path.join(models_dir, 'best_transformer_model.pt')
    
    if not all(os.path.exists(path) for path in [lstm_model_path, improved_lstm_model_path, transformer_model_path]):
        print("Not all model checkpoints found. Please train all models first.")
        missing = []
        if not os.path.exists(lstm_model_path):
            missing.append("LSTM")
        if not os.path.exists(improved_lstm_model_path):
            missing.append("Improved LSTM")
        if not os.path.exists(transformer_model_path):
            missing.append("Transformer")
        print(f"Missing models: {', '.join(missing)}")
        return
    
    # Initialize models
    # 1. LSTM Model
    lstm_model = LSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['model']['embedding_dim'],
        hidden_dim=CONFIG['model']['hidden_dim'],
        num_layers=CONFIG['model']['num_layers'],
        dropout=CONFIG['model']['dropout']
    ).to(device)
    
    # 2. Improved LSTM Model
    improved_lstm_model = ImprovedLSTMSeq2Seq(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['improved_lstm']['embedding_dim'],
        hidden_dim=CONFIG['improved_lstm']['hidden_dim'],
        num_layers=CONFIG['improved_lstm']['num_layers'],
        dropout=CONFIG['improved_lstm']['dropout'],
        attention_dim=CONFIG['improved_lstm']['attention_dim']
    ).to(device)
    
    # 3. Transformer Model
    transformer_model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=CONFIG['transformer']['embedding_dim'],
        nhead=CONFIG['transformer']['num_heads'],
        num_encoder_layers=CONFIG['transformer']['num_encoder_layers'],
        num_decoder_layers=CONFIG['transformer']['num_decoder_layers'],
        dim_feedforward=CONFIG['transformer']['dim_feedforward'],
        dropout=CONFIG['transformer']['dropout']
    ).to(device)
    
    # Initialize optimizers and criteria
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_idx[tokenizer.PAD_TOKEN])
    
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=CONFIG['model']['learning_rate'])
    improved_lstm_optimizer = optim.Adam(improved_lstm_model.parameters(), lr=CONFIG['model']['learning_rate'])
    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=CONFIG['model']['learning_rate'])
    
    # Initialize trainers
    lstm_trainer = LSTMTrainer(
        lstm_model, lstm_optimizer, criterion, tokenizer, CONFIG, device
    )
    
    improved_lstm_trainer = ImprovedLSTMTrainer(
        improved_lstm_model, improved_lstm_optimizer, criterion, tokenizer, CONFIG, device
    )
    
    transformer_trainer = TransformerTrainer(
        transformer_model, transformer_optimizer, criterion, tokenizer, CONFIG, device
    )
    
    # Load trained models
    print("Loading trained models from checkpoints...")
    lstm_trainer.load_model('best_lstm_model.pt')
    improved_lstm_trainer.load_model('best_improved_lstm_model.pt')
    transformer_trainer.load_model('best_transformer_model.pt')
    
    # Get test data
    data_processor = DataProcessor(CONFIG, device)
    processed_data = data_processor.process_data(functions, expansions)
    test_funcs = processed_data['test_funcs']
    
    # Select a subset of test functions for evaluation to save time
    num_examples = len(test_funcs)
    print(f"Using {num_examples} examples for model comparison")
    test_funcs_subset = test_funcs[:num_examples]
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        lstm_trainer, improved_lstm_trainer, transformer_trainer, tokenizer, CONFIG
    )
    
    # Compare models
    results, metrics = evaluator.compare_models(test_funcs_subset)
    
    print("Model comparison completed!")
    
    # Print a few examples
    print("\nSample Predictions:")
    for i in range(min(5, len(results))):
        print(f"\nFunction: {results[i]['function']}")
        print(f"True Expansion: {results[i]['true_expansion']}")
        print(f"LSTM: {results[i]['lstm_prediction']}")
        print(f"Improved LSTM: {results[i]['improved_lstm_prediction']}")
        print(f"Transformer: {results[i]['transformer_prediction']}")


if __name__ == "__main__":
    main()
