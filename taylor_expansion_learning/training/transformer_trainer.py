"""
Transformer Trainer Module

This module contains the trainer class for the Transformer sequence-to-sequence model.
"""

import os
import math
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class TransformerTrainer:
    """
    Handles training and evaluation of the Transformer model.
    
    This class manages the training process, model evaluation,
    and prediction functionality for the Transformer model.
    
    Attributes:
        model (TransformerSeq2Seq): Transformer sequence-to-sequence model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        tokenizer (Tokenizer): Tokenizer for processing expressions.
        config (dict): Configuration parameters.
        best_val_loss (float): Best validation loss achieved.
        device (torch.device): Device to use for tensor operations.
        pad_idx (int): Padding token index.
    """
    
    def __init__(self, model, optimizer, criterion, tokenizer, config,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize the Transformer trainer.
        
        Args:
            model (TransformerSeq2Seq): Transformer sequence-to-sequence model.
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
        self.pad_idx = tokenizer.token_to_idx[tokenizer.PAD_TOKEN]

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
        num_batches = 0

        for i, batch in enumerate(tqdm(data_loader, desc="Training Transformer")):
            # Check batch size - skip if not matching expected size
            if batch['functions'].size(0) != self.config['model']['batch_size']:
                print(f"Skipping batch with size {batch['functions'].size(0)}")
                continue

            self.optimizer.zero_grad()

            src = batch['functions']
            trg = batch['input']  # Input to decoder
            trg_y = batch['target']  # Target for decoder

            # Forward pass
            output = self.model(src, trg, self.pad_idx)

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
            num_batches += 1

        return epoch_loss / max(num_batches, 1)  # Avoid division by zero

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
        num_batches = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader, desc="Evaluating Transformer")):
                # Check batch size - skip if not matching expected size
                if batch['functions'].size(0) != self.config['model']['batch_size']:
                    print(f"Skipping validation batch with size {batch['functions'].size(0)}")
                    continue

                src = batch['functions']
                trg = batch['input']
                trg_y = batch['target']

                # Forward pass
                output = self.model(src, trg, self.pad_idx)

                # Reshape for loss calculation
                output_flat = output.contiguous().view(-1, output.shape[-1])
                trg_y_flat = trg_y.contiguous().view(-1)

                # Calculate loss
                loss = self.criterion(output_flat, trg_y_flat)

                epoch_loss += loss.item()
                num_batches += 1

        return epoch_loss / max(num_batches, 1)  # Avoid division by zero

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
        print(f"Starting Transformer model training for {epochs} epochs...")

        # Ensure we're using consistent batch sizes
        batch_size = next(iter(train_loader))['functions'].size(0)
        print(f"Training with batch size: {batch_size}")

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
                self.save_model('best_transformer_model.pt')
                print("Saved best model.")

            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(f'transformer_model_epoch_{epoch+1}.pt')

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

        # Start with SOS token
        output_tokens = [self.tokenizer.token_to_idx[self.tokenizer.SOS_TOKEN]]

        # Get the padding token index
        pad_idx = self.tokenizer.token_to_idx[self.tokenizer.PAD_TOKEN]

        with torch.no_grad():
            # Encode the source function
            src_embedded = self.model.embedding(function_tensor) * math.sqrt(self.model.d_model)
            src_embedded = self.model.pos_encoder(src_embedded)
            src_padding_mask = self.model._create_padding_mask(function_tensor, pad_idx)

            memory = self.model.transformer.encoder(
                src=src_embedded,
                src_key_padding_mask=src_padding_mask
            )

            for _ in range(max_length):
                # Create tensor from current output tokens
                trg_tensor = torch.tensor([output_tokens], dtype=torch.long).to(self.device)

                # Create padding mask for target
                trg_padding_mask = self.model._create_padding_mask(trg_tensor, pad_idx)

                # Create target mask to prevent attending to future tokens
                trg_mask = self.model._generate_square_subsequent_mask(trg_tensor.size(1))

                # Embed and add positional encoding
                trg_embedded = self.model.embedding(trg_tensor) * math.sqrt(self.model.d_model)
                trg_embedded = self.model.pos_encoder(trg_embedded)

                # Decode
                output = self.model.transformer.decoder(
                    tgt=trg_embedded,
                    memory=memory,
                    tgt_mask=trg_mask,
                    tgt_key_padding_mask=trg_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )

                # Final linear layer
                output = self.model.fc_out(output)

                # Get the most likely next token
                pred_token = output[0, -1, :].argmax().item()
                output_tokens.append(pred_token)

                # Stop if EOS token is generated
                if pred_token == self.tokenizer.token_to_idx[self.tokenizer.EOS_TOKEN]:
                    break

        # Convert tokens back to text
        expansion_text = self.tokenizer.decode(output_tokens)

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
        plt.title('Transformer Model Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Save figure
        plot_path = os.path.join(self.config['paths']['results_dir'], 'transformer_losses.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Loss plot saved to {plot_path}")
