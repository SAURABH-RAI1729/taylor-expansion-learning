"""
Tokenizer Module for Mathematical Expressions

This module provides functionality to tokenize mathematical expressions into
sequences suitable for neural network processing.
"""

import json


class Tokenizer:
    """
    Tokenizer for mathematical expressions.
    
    This class handles the tokenization of mathematical expressions into
    sequences of tokens, building a vocabulary, and converting between
    tokens and indices.
    
    Attributes:
        token_to_idx (dict): Mapping from tokens to indices.
        idx_to_token (dict): Mapping from indices to tokens.
        vocab_size (int): Size of the vocabulary.
        PAD_TOKEN (str): Special token for padding.
        UNK_TOKEN (str): Special token for unknown tokens.
        SOS_TOKEN (str): Special token for start of sequence.
        EOS_TOKEN (str): Special token for end of sequence.
    """
    
    def __init__(self):
        """Initialize the tokenizer with special tokens and common math symbols."""
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_size = 0

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'  # Start of sequence
        self.EOS_TOKEN = '<EOS>'  # End of sequence

        # Initialize with special tokens
        self._add_token(self.PAD_TOKEN)
        self._add_token(self.UNK_TOKEN)
        self._add_token(self.SOS_TOKEN)
        self._add_token(self.EOS_TOKEN)

        # Add common mathematical tokens
        common_tokens = [
            '+', '-', '*', '/', '^', '**', '(', ')', '[', ']',
            'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'atan', 'sinh', 'cosh',
            'x', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
            'pi', 'e'
        ]
        for token in common_tokens:
            self._add_token(token)

    def _add_token(self, token):
        """
        Add a token to the vocabulary if it doesn't exist.
        
        Args:
            token (str): Token to add to the vocabulary.
        """
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.vocab_size
            self.idx_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def build_vocab(self, texts):
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts (list): List of strings to extract tokens from.
        """
        print("Building vocabulary...")
        for text in texts:
            tokens = self._tokenize_text(text)
            for token in tokens:
                self._add_token(token)
        print(f"Vocabulary size: {self.vocab_size}")

    def _tokenize_text(self, text):
        """
        Simple tokenization for mathematical expressions.
        
        Args:
            text (str): Mathematical expression as a string.
            
        Returns:
            list: List of tokens.
        """
        # Handle common functions
        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'atan', 'sinh', 'cosh']:
            text = text.replace(func, f" {func} ")

        # Handle operators and parentheses
        for char in ['+', '-', '*', '/', '^', '**', '(', ')', '[', ']', ',']:
            text = text.replace(char, f" {char} ")

        # Split by whitespace and filter empty tokens
        tokens = [token for token in text.split() if token]
        return tokens

    def tokenize(self, text, max_length=None):
        """
        Tokenize text to indices with optional padding.
        
        Args:
            text (str): Text to tokenize.
            max_length (int, optional): Maximum sequence length.
            
        Returns:
            list: List of token indices.
        """
        tokens = [self.SOS_TOKEN] + self._tokenize_text(text) + [self.EOS_TOKEN]

        # Convert tokens to indices
        indices = [self.token_to_idx.get(token, self.token_to_idx[self.UNK_TOKEN]) for token in tokens]

        if max_length is not None:
            # Truncate if necessary
            if len(indices) > max_length:
                indices = indices[:max_length]

            # Pad if necessary
            padding_length = max_length - len(indices)
            if padding_length > 0:
                indices = indices + [self.token_to_idx[self.PAD_TOKEN]] * padding_length

        return indices

    def decode(self, indices):
        """
        Convert indices back to text.
        
        Args:
            indices (list): List of token indices.
            
        Returns:
            str: Decoded text.
        """
        tokens = [self.idx_to_token.get(idx, self.UNK_TOKEN) for idx in indices]

        # Remove special tokens
        tokens = [token for token in tokens if token not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]]

        # Join tokens
        text = ' '.join(tokens)
        return text

    def save(self, path):
        """
        Save tokenizer to file.
        
        Args:
            path (str): Path to save tokenizer.
        """
        data = {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': {int(k): v for k, v in self.idx_to_token.items()},
            'vocab_size': self.vocab_size
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        """
        Load tokenizer from file.
        
        Args:
            path (str): Path to load tokenizer from.
            
        Returns:
            Tokenizer: Loaded tokenizer.
        """
        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_idx = data['token_to_idx']
        tokenizer.idx_to_token = {int(k): v for k, v in data['idx_to_token'].items()}
        tokenizer.vocab_size = data['vocab_size']

        return tokenizer
