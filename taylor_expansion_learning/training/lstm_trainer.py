"""
LSTM Trainer Module

This module contains the trainer class for the LSTM sequence-to-sequence model.
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


class LSTMTrainer:
    """
    Handles training and evaluation of the LSTM model.
    
    This class manages the training process, model evaluation,
    and prediction functionality for the LSTM model.
    
    Attributes:
        model (LSTMSeq2Seq): LSTM sequence-to-sequence model.
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
        Initialize the LSTM trainer.
        
        Args:
            model (LSTMSeq2Seq): LSTM sequence-to-sequence model.
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

        for i, batch in enumerate(tqdm(data_loader, desc="Training LSTM")):
            self.optimizer.zero_grad()

            src = batch['functions']
            trg = batch['input']  # Input to decoder
            trg_y = batch['target']  # Target for decoder

            # Forward pass
            output = self.model(src, trg)

            # Reshape for loss calculation
            output_flat = output.contiguous().view(-1, output.shape[-1])
            trg_y_flat = trg_y.contiguous().view(-1)

            # Calculate loss
            loss = self.criterion(output_flat, trg_y_flat)

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
            for i, batch in enumerate(tqdm(data_loader, desc="Evaluating LSTM")):
                src = batch['functions']
                trg = batch['input']
                trg_y = batch['target']

                # Forward pass
                output = self.model(src, trg)

                # Reshape for loss calculation
                output_flat = output.contiguous().view(-1, output.shape[-1])
                trg_y_flat = trg_y.contiguous().view(-1)

                # Calculate loss
                loss = self.criterion(output_flat, trg_y_flat)

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
        print(f"Starting LSTM model training for {epochs} epochs...")

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
                self.save_model('best_lstm_model.pt')
                print("Saved best model.")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f'lstm_model_epoch_{epoch+1}.pt')

        # Plot losses
        self.plot_losses(train_losses, val_losses)

        print("Training complete.")
        return train_losses, val_losses

    def predict(self, function_text, max_length=100):
        """
        Generate Taylor expansion for a given function.
        
        Args:
            function_text (str): Function expression as a string.
            max_length (int): Maximum length of the generated sequence.
            
        Returns:
            str: Generated Taylor expansion.
        """
        self.model.eval()

        # Tokenize function
        function_tokens = self.tokenizer.tokenize(function_text)
        function_tensor = torch.tensor(function_tokens, dtype=torch.long).unsqueeze(0).to(self.device)

        # Encode function
        _, (hidden, cell) = self.model.encoder(function_tensor)

        # Start with SOS token
        input_token = torch.tensor([[self.tokenizer.token_to_idx[self.tokenizer.SOS_TOKEN]]], device=self.device)

        # Store generated tokens
        generated_tokens = [self.tokenizer.token_to_idx[self.tokenizer.SOS_TOKEN]]

        with torch.no_grad():
            for _ in range(max_length):
                # Decode one step
                output, hidden, cell = self.model.decoder(input_token, hidden, cell)

                # Get the most likely next token
                pred_token = output.argmax(2).item()
                generated_tokens.append(pred_token)

                # Stop if EOS token is generated
                if pred_token == self.tokenizer.token_to_idx[self.tokenizer.EOS_TOKEN]:
                    break

                # Use the predicted token as the next input
                input_token = torch.tensor([[pred_token]], device=self.device)

        # Convert tokens back to text
        expansion_text = self.tokenizer.decode(generated_tokens)

        return expansion_text

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
        plt.title('LSTM Model Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(self.config['paths']['results_dir'], 'lstm_losses.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Loss plot saved to {plot_path}")
