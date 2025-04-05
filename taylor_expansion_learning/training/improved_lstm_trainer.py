"""
Improved LSTM Trainer Module

This module contains the trainer class for the improved LSTM sequence-to-sequence
model with attention mechanism.
"""

import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImprovedLSTMTrainer:
    """
    Handles training and evaluation of the improved LSTM model with attention.
    
    This class manages the training process, model evaluation,
    visualization of attention, and prediction functionality.
    
    Attributes:
        model (ImprovedLSTMSeq2Seq): Improved LSTM sequence-to-sequence model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        tokenizer (Tokenizer): Tokenizer for processing expressions.
        config (dict): Configuration parameters.
        best_val_loss (float): Best validation loss achieved.
        device (torch.device): Device to use for tensor operations.
    """
    
    def __init__(self, model, optimizer, criterion, tokenizer, config,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the improved LSTM trainer.
        
        Args:
            model (ImprovedLSTMSeq2Seq): Improved LSTM sequence-to-sequence model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            tokenizer (Tokenizer): Tokenizer for processing expressions.
            config (dict): Configuration parameters.
            device (torch.device): Device to use for tensor operations.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.config = config
        self.best_val_loss = float('inf')
        self.device = device

    def train_epoch(self, data_loader):
        """
        Train for one epoch.
        
        Args:
            data_loader (DataLoader): Data loader for training.
            
        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        epoch_loss = 0

        for i, batch in enumerate(tqdm(data_loader, desc="Training Improved LSTM")):
            self.optimizer.zero_grad()

            src = batch['functions']
            trg = batch['input']  # Input to decoder
            trg_y = batch['target']  # Target for decoder

            # Forward pass
            outputs, _ = self.model(src, trg)

            # Reshape for loss calculation
            outputs_flat = outputs.contiguous().view(-1, outputs.shape[-1])
            trg_y_flat = trg_y.contiguous().view(-1)

            # Calculate loss
            loss = self.criterion(outputs_flat, trg_y_flat)

            # Backward pass and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def evaluate(self, data_loader):
        """
        Evaluate the model.
        
        Args:
            data_loader (DataLoader): Data loader for evaluation.
            
        Returns:
            float: Average loss for the evaluation.
        """
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Evaluating Improved LSTM")):
                src = batch['functions']
                trg = batch['input']
                trg_y = batch['target']

                # Forward pass
                outputs, _ = self.model(src, trg)

                # Reshape for loss calculation
                outputs_flat = outputs.contiguous().view(-1, outputs.shape[-1])
                trg_y_flat = trg_y.contiguous().view(-1)

                # Calculate loss
                loss = self.criterion(outputs_flat, trg_y_flat)

                epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def train(self, train_loader, val_loader, epochs):
        """
        Train the model for a number of epochs.
        
        Args:
            train_loader (DataLoader): Data loader for training.
            val_loader (DataLoader): Data loader for validation.
            epochs (int): Number of epochs to train.
            
        Returns:
            tuple: Lists of training and validation losses.
        """
        print(f"Starting Improved LSTM model training for {epochs} epochs...")

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)

            # Evaluate
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)

            print(f"Epoch: {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model('best_improved_lstm_model.pt')
                print("Saved best model.")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f'improved_lstm_model_epoch_{epoch+1}.pt')

        # Plot losses
        self.plot_losses(train_losses, val_losses)

        print("Training complete.")
        return train_losses, val_losses

    def predict(self, function_text, max_length=100, visualize_attention=False):
        """
        Generate Taylor expansion for a given function with optional attention visualization.
        
        Args:
            function_text (str): Function expression as a string.
            max_length (int): Maximum length of the generated sequence.
            visualize_attention (bool): Whether to visualize attention weights.
            
        Returns:
            str: Generated Taylor expansion.
        """
        self.model.eval()

        # Tokenize function
        function_tokens = self.tokenizer.tokenize(function_text)
        function_tensor = torch.tensor(function_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        # Encode function
        encoder_outputs, (hidden, cell) = self.model.encoder(function_tensor)

        # Start with SOS token
        input_token = torch.tensor([[self.tokenizer.token_to_idx[self.tokenizer.SOS_TOKEN]]], device=self.device)

        # Store generated tokens and attention weights
        generated_tokens = [self.tokenizer.token_to_idx[self.tokenizer.SOS_TOKEN]]
        attentions = []

        with torch.no_grad():
            for _ in range(max_length):
                # Decode one step with attention
                output, hidden, cell, attention = self.model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )

                # Get the most likely next token
                pred_token = output.argmax(1).item()
                generated_tokens.append(pred_token)

                # Store attention weights for visualization
                if visualize_attention:
                    attentions.append(attention.cpu().numpy())

                # Stop if EOS token is generated
                if pred_token == self.tokenizer.token_to_idx[self.tokenizer.EOS_TOKEN]:
                    break

                # Use the predicted token as the next input
                input_token = torch.tensor([[pred_token]], device=self.device)

        # Convert tokens back to text
        expansion_text = self.tokenizer.decode(generated_tokens)

        # Visualize attention if requested
        if visualize_attention and attentions:
            self._visualize_attention(function_text, expansion_text, attentions)

        return expansion_text

    def _visualize_attention(self, function_text, expansion_text, attentions):
        """
        Visualize attention weights.
        
        Args:
            function_text (str): Source function expression.
            expansion_text (str): Generated expansion.
            attentions (list): List of attention weight matrices.
        """
        # Tokenize for visualization
        function_tokens = ['<SOS>'] + self.tokenizer._tokenize_text(function_text) + ['<EOS>']
        expansion_tokens = self.tokenizer._tokenize_text(expansion_text)

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        # Create attention heatmap
        attention_matrix = torch.cat([torch.tensor(a) for a in attentions], dim=0)
        cax = ax.matshow(attention_matrix, cmap='viridis')

        # Add colorbar
        fig.colorbar(cax)

        # Set axis labels
        ax.set_xticklabels([''] + function_tokens, rotation=90)
        ax.set_yticklabels([''] + expansion_tokens)

        # Set axis titles
        ax.set_xlabel('Function Tokens')
        ax.set_ylabel('Expansion Tokens')
        ax.xaxis.set_label_position('top')

        # Save figure
        attention_plot_path = os.path.join(
            self.config['paths']['results_dir'], 
            'attention_visualization.png'
        )
        plt.savefig(attention_plot_path)
        plt.close()

        print(f"Attention visualization saved to {attention_plot_path}")

    def save_model(self, file_name):
        """
        Save model checkpoint.
        
        Args:
            file_name (str): Name of the checkpoint file.
        """
        model_path = os.path.join(self.config['paths']['models_dir'], file_name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, model_path)

    def load_model(self, file_name):
        """
        Load model checkpoint.
        
        Args:
            file_name (str): Name of the checkpoint file.
            
        Returns:
            dict: Configuration from the checkpoint.
        """
        model_path = os.path.join(self.config['paths']['models_dir'], file_name)
        checkpoint = torch.load(model_path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

        return checkpoint['config']

    def plot_losses(self, train_losses, val_losses):
        """
        Plot training and validation losses.
        
        Args:
            train_losses (list): List of training losses.
            val_losses (list): List of validation losses.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Improved LSTM Model Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(self.config['paths']['results_dir'], 'improved_lstm_losses.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Loss plot saved to {plot_path}")
