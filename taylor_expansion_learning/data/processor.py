"""
Data Processor Module

This module contains functionality for preprocessing and preparing data
for training machine learning models on Taylor expansions.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from taylor_expansion_learning.data.tokenizer import Tokenizer


class TaylorDataset(Dataset):
    """
    Dataset class for Taylor expansion data.
    
    This class handles the storage and retrieval of tokenized
    function-expansion pairs.
    
    Attributes:
        functions (list): List of function expressions.
        expansions (list): List of Taylor expansions.
        tokenizer (Tokenizer): Tokenizer for processing expressions.
    """
    
    def __init__(self, functions, expansions, tokenizer):
        """
        Initialize the dataset.
        
        Args:
            functions (list): List of function strings.
            expansions (list): List of expansion strings.
            tokenizer (Tokenizer): Tokenizer for processing expressions.
        """
        self.functions = functions
        self.expansions = expansions
        self.tokenizer = tokenizer

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.functions)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            dict: Dictionary containing tokenized function and expansion.
        """
        function = self.functions[idx]
        expansion = self.expansions[idx]

        # Tokenize
        func_tokens = self.tokenizer.tokenize(function)
        exp_tokens = self.tokenizer.tokenize(expansion)

        # Convert to tensor
        func_tensor = torch.tensor(func_tokens, dtype=torch.long)
        exp_tensor = torch.tensor(exp_tokens, dtype=torch.long)

        return {
            'function': func_tensor,
            'expansion': exp_tensor,
            'function_text': function,
            'expansion_text': expansion
        }


class DataProcessor:
    """
    Handles data processing tasks.
    
    This class is responsible for processing raw function-expansion pairs
    into a format suitable for training machine learning models.
    
    Attributes:
        config (dict): Configuration parameters.
        tokenizer (Tokenizer): Tokenizer for processing expressions.
        device (torch.device): Device to use for tensor operations.
    """
    
    def __init__(self, config, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the data processor.
        
        Args:
            config (dict): Configuration parameters.
            device (torch.device): Device to use for tensor operations.
        """
        self.config = config
        self.tokenizer = Tokenizer()
        self.device = device

    def process_data(self, functions, expansions):
        """
        Process the data for model training.
        
        Args:
            functions (list): List of function strings.
            expansions (list): List of expansion strings.
            
        Returns:
            dict: Dictionary containing processed data.
        """
        print("Processing data...")

        # Build vocabulary
        self.tokenizer.build_vocab(functions + expansions)

        # Save tokenizer
        tokenizer_path = os.path.join(self.config['paths']['data_dir'], 'tokenizer.json')
        self.tokenizer.save(tokenizer_path)

        # Ensure dataset size is divisible by batch_size to avoid issues
        batch_size = self.config['model']['batch_size']
        dataset_size = len(functions)
        usable_size = (dataset_size // batch_size) * batch_size

        print(f"Original dataset size: {dataset_size}")
        print(f"Usable size (multiple of batch_size {batch_size}): {usable_size}")

        # Use only the portion of the dataset that's divisible by batch_size
        functions = functions[:usable_size]
        expansions = expansions[:usable_size]

        # Split data
        train_funcs, temp_funcs, train_exps, temp_exps = train_test_split(
            functions, expansions, test_size=0.3, random_state=42
        )

        # Split the remaining data into validation and test sets
        val_funcs, test_funcs, val_exps, test_exps = train_test_split(
            temp_funcs, temp_exps, test_size=0.5, random_state=42
        )

        # Create datasets
        train_dataset = TaylorDataset(train_funcs, train_exps, self.tokenizer)
        val_dataset = TaylorDataset(val_funcs, val_exps, self.tokenizer)
        test_dataset = TaylorDataset(test_funcs, test_exps, self.tokenizer)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=True,
            collate_fn=self.collate_fn,
            drop_last=True  # Drop the last batch if it's smaller than batch_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True  # Drop the last batch if it's smaller than batch_size
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['model']['batch_size'],
            shuffle=False,
            collate_fn=self.collate_fn,
            drop_last=True
        )

        print(f"Data processing complete. Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
        print(f"Test size: {len(test_dataset)}")

        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'tokenizer': self.tokenizer,
            'vocab_size': self.tokenizer.vocab_size,
            'test_funcs': test_funcs,
            'test_exps': test_exps
        }

    def collate_fn(self, batch):
        """
        Custom collate function for handling variable length sequences.
        
        Args:
            batch (list): List of samples from the dataset.
            
        Returns:
            dict: Batch of processed data.
        """
        # Verify batch size and log warning if it's unexpected
        if len(batch) != self.config['model']['batch_size']:
            print(f"Warning: Received batch of size {len(batch)}, expected {self.config['model']['batch_size']}")

        # Extract items
        functions = [item['function'] for item in batch]
        expansions = [item['expansion'] for item in batch]
        function_texts = [item['function_text'] for item in batch]
        expansion_texts = [item['expansion_text'] for item in batch]

        # Get sequence lengths
        func_lengths = [len(f) for f in functions]
        exp_lengths = [len(e) for e in expansions]

        # Pad sequences
        max_func_len = min(max(func_lengths), self.config['tokenization']['max_function_length'])
        max_exp_len = min(max(exp_lengths), self.config['tokenization']['max_expansion_length'])

        padded_functions = []
        padded_expansions = []

        for f in functions:
            if len(f) > max_func_len:
                padded_functions.append(f[:max_func_len])
            else:
                padding = torch.tensor([self.tokenizer.token_to_idx[self.tokenizer.PAD_TOKEN]] * (max_func_len - len(f)), dtype=torch.long)
                padded_functions.append(torch.cat([f, padding]))

        for e in expansions:
            if len(e) > max_exp_len:
                padded_expansions.append(e[:max_exp_len])
            else:
                padding = torch.tensor([self.tokenizer.token_to_idx[self.tokenizer.PAD_TOKEN]] * (max_exp_len - len(e)), dtype=torch.long)
                padded_expansions.append(torch.cat([e, padding]))

        # Stack tensors and ensure they're Long tensors for embedding layers
        functions_tensor = torch.stack(padded_functions).long()
        expansions_tensor = torch.stack(padded_expansions).long()

        # Create input and target tensors for training
        input_tensor = expansions_tensor[:, :-1].long()  # Remove last token (EOS)
        target_tensor = expansions_tensor[:, 1:].long()  # Remove first token (SOS)

        return {
            'functions': functions_tensor.to(self.device),
            'expansions': expansions_tensor.to(self.device),
            'input': input_tensor.to(self.device),
            'target': target_tensor.to(self.device),
            'function_texts': function_texts,
            'expansion_texts': expansion_texts
        }
