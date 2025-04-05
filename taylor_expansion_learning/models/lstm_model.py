"""
LSTM Model Module

This module contains the LSTM-based sequence-to-sequence model implementation
for learning Taylor expansions.
"""

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM Encoder for encoding the function expressions.
    
    This encoder transforms input sequences into a hidden state representation.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer.
        lstm (nn.LSTM): LSTM layers.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the LSTM encoder.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            tuple: LSTM outputs for each timestep and final hidden states
        """
        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class LSTMDecoder(nn.Module):
    """
    LSTM Decoder for generating Taylor expansions.
    
    This decoder transforms hidden states back into sequences.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer.
        lstm (nn.LSTM): LSTM layers.
        fc_out (nn.Linear): Output layer.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the LSTM decoder.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        """
        Forward pass through the decoder.
        
        Args:
            x: Input tensor of shape (batch_size, 1)
            hidden: Hidden state from encoder or previous decoder step
            cell: Cell state from encoder or previous decoder step
            
        Returns:
            tuple: Prediction for the next token and updated hidden states
        """
        embedded = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class LSTMSeq2Seq(nn.Module):
    """
    Sequence-to-sequence model with LSTM for learning Taylor expansions.
    
    This model connects an encoder and decoder for sequence-to-sequence learning.
    
    Attributes:
        encoder (LSTMEncoder): Encoder module.
        decoder (LSTMDecoder): Decoder module.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the sequence-to-sequence model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(LSTMSeq2Seq, self).__init__()
        self.encoder = LSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, trg):
        """
        Forward pass through the sequence-to-sequence model.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            
        Returns:
            torch.Tensor: Predictions for each token in the target sequence
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        # Encoder
        _, (hidden, cell) = self.encoder(src)

        # Initialize decoder input with SOS token
        outputs = torch.zeros(batch_size, trg_len, self.decoder.fc_out.out_features).to(src.device)

        # Decoder forward pass
        for t in range(trg_len):
            decoder_input = trg[:, t].unsqueeze(1)  # (batch_size, 1)
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)

        return outputs
