"""
Transformer Model Module

This module contains the Transformer-based sequence-to-sequence model
implementation for learning Taylor expansions.
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    
    This module adds positional information to the input embeddings
    to help the model understand the order of tokens in the sequence.
    
    Attributes:
        pe (torch.Tensor): Precomputed positional encodings.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): Dimension of the embeddings.
            max_len (int): Maximum sequence length to precompute.
        """
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encoding added
        """
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:x.size(1), :]


class TransformerSeq2Seq(nn.Module):
    """
    Transformer Sequence-to-Sequence model for Taylor expansions.
    
    This model uses the Transformer architecture for sequence-to-sequence learning.
    
    Attributes:
        embedding (nn.Embedding): Token embedding layer.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        transformer (nn.Transformer): Transformer model.
        fc_out (nn.Linear): Output projection layer.
        d_model (int): Dimension of the model.
    """
    
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, dropout=0.1):
        """
        Initialize the Transformer sequence-to-sequence model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model.
            nhead (int): Number of attention heads.
            num_encoder_layers (int): Number of encoder layers.
            num_decoder_layers (int): Number of decoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout rate.
        """
        super(TransformerSeq2Seq, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Final output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Model parameters
        self.d_model = d_model

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for decoder self-attention.
        
        Args:
            sz (int): Size of the square mask.
            
        Returns:
            torch.Tensor: Mask tensor
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask.to(next(self.parameters()).device)

    def _create_padding_mask(self, x, pad_idx):
        """
        Create padding mask for transformer.
        
        Args:
            x (torch.Tensor): Input tensor.
            pad_idx (int): Padding token index.
            
        Returns:
            torch.Tensor: Boolean mask tensor
        """
        return x == pad_idx

    def forward(self, src, trg, src_pad_idx=0):
        """
        Forward pass through the Transformer model.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            src_pad_idx: Padding index for source sequence
            
        Returns:
            torch.Tensor: Output predictions
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        # Create source and target padding masks
        src_pad_mask = self._create_padding_mask(src, src_pad_idx)  # [batch_size, src_len]
        trg_pad_mask = self._create_padding_mask(trg, src_pad_idx)  # [batch_size, trg_len]

        # Create target mask to prevent attending to future tokens
        trg_mask = self._generate_square_subsequent_mask(trg.size(1))  # [trg_len, trg_len]

        # Embed source and target
        src_emb = self.embedding(src) * math.sqrt(self.d_model)  # [batch_size, src_len, d_model]
        trg_emb = self.embedding(trg) * math.sqrt(self.d_model)  # [batch_size, trg_len, d_model]

        # Add positional encoding
        src_emb = self.pos_encoder(src_emb)  # [batch_size, src_len, d_model]
        trg_emb = self.pos_encoder(trg_emb)  # [batch_size, trg_len, d_model]

        # Transformer forward pass
        output = self.transformer(
            src=src_emb,
            tgt=trg_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=trg_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=trg_mask
        )  # [batch_size, trg_len, d_model]

        # Final linear layer
        output = self.fc_out(output)  # [batch_size, trg_len, vocab_size]

        return output
