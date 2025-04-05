"""
Improved LSTM Model Module

This module contains an enhanced LSTM-based sequence-to-sequence model
with attention mechanism for learning Taylor expansions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention mechanism for the improved LSTM model.
    
    This module implements the attention mechanism that allows the decoder
    to focus on different parts of the encoder output at each decoding step.
    
    Attributes:
        attn (nn.Linear): Linear layer for attention scores.
        v (nn.Linear): Linear layer for attention weights.
    """
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        Initialize the attention mechanism.
        
        Args:
            encoder_dim (int): Dimension of the encoder hidden states.
            decoder_dim (int): Dimension of the decoder hidden states.
            attention_dim (int): Dimension of the attention mechanism.
        """
        super(Attention, self).__init__()
        
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Forward pass through the attention mechanism.
        
        Args:
            encoder_outputs: Outputs from the encoder (batch_size, src_len, encoder_dim)
            decoder_hidden: Current decoder hidden state (batch_size, decoder_dim)
            
        Returns:
            tuple: Attention weights and context vector
        """
        # encoder_outputs: [batch_size, src_len, encoder_dim]
        # decoder_hidden: [batch_size, decoder_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Expand decoder hidden state for attention calculation
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate attention energy
        encoder_transformed = self.encoder_attn(encoder_outputs)  # [batch_size, src_len, attention_dim]
        decoder_transformed = self.decoder_attn(decoder_hidden)   # [batch_size, src_len, attention_dim]
        
        # Apply tanh activation and compute attention weights
        energy = torch.tanh(encoder_transformed + decoder_transformed)  # [batch_size, src_len, attention_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # [batch_size, src_len]
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, encoder_dim]
        context = context.squeeze(1)  # [batch_size, encoder_dim]
        
        return attention_weights, context


class ImprovedLSTMEncoder(nn.Module):
    """
    Improved LSTM Encoder with bidirectional LSTM.
    
    This encoder transforms input sequences into a hidden state representation
    using a bidirectional LSTM for better context capturing.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer.
        lstm (nn.LSTM): Bidirectional LSTM layers.
        fc_hidden (nn.Linear): Linear layer for hidden state transformation.
        fc_cell (nn.Linear): Linear layer for cell state transformation.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        """
        Initialize the improved LSTM encoder.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        super(ImprovedLSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Project bidirectional outputs to decoder dimensions
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the improved encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            tuple: LSTM outputs for each timestep and transformed final hidden states
        """
        # x: [batch_size, src_len]
        
        embedded = self.dropout(self.embedding(x))  # [batch_size, src_len, embedding_dim]
        
        # Pass through bidirectional LSTM
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden: [num_layers * 2, batch_size, hidden_dim // 2]
        # cell: [num_layers * 2, batch_size, hidden_dim // 2]
        
        # Concatenate the forward and backward hidden states
        hidden = self._transform_hidden(hidden)  # [num_layers, batch_size, hidden_dim]
        cell = self._transform_cell(cell)        # [num_layers, batch_size, hidden_dim]
        
        return outputs, (hidden, cell)
    
    def _transform_hidden(self, hidden):
        """
        Transform bidirectional hidden states for the decoder.
        
        Args:
            hidden: Bidirectional hidden states
            
        Returns:
            torch.Tensor: Transformed hidden states
        """
        # hidden: [num_layers * 2, batch_size, hidden_dim // 2]
        
        # Reshape to separate layers and directions
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim // 2)
        
        # Concatenate forward and backward states
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)  # [num_layers, batch_size, hidden_dim]
        
        # Apply transformation
        return self.fc_hidden(hidden)
    
    def _transform_cell(self, cell):
        """
        Transform bidirectional cell states for the decoder.
        
        Args:
            cell: Bidirectional cell states
            
        Returns:
            torch.Tensor: Transformed cell states
        """
        # cell: [num_layers * 2, batch_size, hidden_dim // 2]
        
        # Reshape to separate layers and directions
        cell = cell.view(self.num_layers, 2, -1, self.hidden_dim // 2)
        
        # Concatenate forward and backward states
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)  # [num_layers, batch_size, hidden_dim]
        
        # Apply transformation
        return self.fc_cell(cell)


class ImprovedLSTMDecoder(nn.Module):
    """
    Improved LSTM Decoder with attention mechanism.
    
    This decoder transforms hidden states back into sequences with
    the help of attention on encoder outputs.
    
    Attributes:
        embedding (nn.Embedding): Embedding layer.
        attention (Attention): Attention mechanism.
        lstm (nn.LSTM): LSTM layers.
        fc_out (nn.Linear): Output layer.
        dropout (nn.Dropout): Dropout layer.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, attention_dim):
        """
        Initialize the improved LSTM decoder.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            attention_dim (int): Dimension of the attention mechanism.
        """
        super(ImprovedLSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = Attention(hidden_dim, hidden_dim, attention_dim)
        
        # LSTM input: embedding + context vector from attention
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim,  # Include context vector
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer combines hidden state and context vector
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, cell, encoder_outputs):
        """
        Forward pass through the improved decoder.
        
        Args:
            x: Input tensor of shape (batch_size, 1)
            hidden: Hidden state from encoder or previous decoder step
            cell: Cell state from encoder or previous decoder step
            encoder_outputs: Outputs from the encoder for attention
            
        Returns:
            tuple: Prediction for the next token, updated hidden states, and attention weights
        """
        # x: [batch_size, 1]
        # hidden: [num_layers, batch_size, hidden_dim]
        # cell: [num_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        embedded = self.dropout(self.embedding(x))  # [batch_size, 1, embedding_dim]
        
        # Use the top layer of hidden state for attention
        top_hidden = hidden[-1]  # [batch_size, hidden_dim]
        
        # Calculate attention weights and context vector
        attention_weights, context = self.attention(encoder_outputs, top_hidden)
        
        # Expand context vector to match embedded shape
        context = context.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Concatenate embedding and context vector
        lstm_input = torch.cat([embedded, context], dim=2)  # [batch_size, 1, embedding_dim + hidden_dim]
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Get the output from the last layer
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        context = context.squeeze(1)  # [batch_size, hidden_dim]
        
        # Concatenate output and context for prediction
        prediction_input = torch.cat([output, context], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Make prediction
        prediction = self.fc_out(prediction_input)  # [batch_size, vocab_size]
        
        return prediction, hidden, cell, attention_weights


class ImprovedLSTMSeq2Seq(nn.Module):
    """
    Improved sequence-to-sequence model with LSTM and attention.
    
    This model connects an enhanced encoder and decoder with attention
    for better sequence-to-sequence learning.
    
    Attributes:
        encoder (ImprovedLSTMEncoder): Enhanced encoder module.
        decoder (ImprovedLSTMDecoder): Enhanced decoder module.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, attention_dim):
        """
        Initialize the improved sequence-to-sequence model.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the hidden states.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            attention_dim (int): Dimension of the attention mechanism.
        """
        super(ImprovedLSTMSeq2Seq, self).__init__()
        
        self.encoder = ImprovedLSTMEncoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = ImprovedLSTMDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout, attention_dim)
        
    def forward(self, src, trg):
        """
        Forward pass through the improved sequence-to-sequence model.
        
        Args:
            src: Source sequence (batch_size, src_len)
            trg: Target sequence (batch_size, trg_len)
            
        Returns:
            torch.Tensor: Predictions for each token in the target sequence
        """
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        
        # Store outputs for visualization purposes
        outputs = torch.zeros(batch_size, trg_len, self.decoder.vocab_size).to(src.device)
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(src.device)
        
        # Encode the source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src)
        
        # Decoder forward pass with teacher forcing
        for t in range(trg_len):
            decoder_input = trg[:, t].unsqueeze(1)  # [batch_size, 1]
            
            # Pass through decoder with attention
            output, hidden, cell, attention = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            
            # Store outputs and attention weights
            outputs[:, t] = output
            attentions[:, t] = attention
            
        return outputs, attentions
