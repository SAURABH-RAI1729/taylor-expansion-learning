"""
Model Evaluator Module

This module contains functionality for evaluating and comparing
different sequence-to-sequence models for Taylor expansions.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import Levenshtein
from tqdm import tqdm


class ModelEvaluator:
    """
    Compares the performance of different models.
    
    This class handles the comprehensive evaluation and comparison of
    multiple models using various metrics.
    
    Attributes:
        lstm_trainer: Trainer for basic LSTM model.
        improved_lstm_trainer: Trainer for improved LSTM model.
        transformer_trainer: Trainer for Transformer model.
        tokenizer (Tokenizer): Tokenizer for processing expressions.
        config (dict): Configuration parameters.
        models_to_evaluate (list): List of model names to evaluate.
    """
    
    def __init__(self, lstm_trainer, improved_lstm_trainer, transformer_trainer, tokenizer, config):
        """
        Initialize the model evaluator.
        
        Args:
            lstm_trainer: Trainer for basic LSTM model.
            improved_lstm_trainer: Trainer for improved LSTM model.
            transformer_trainer: Trainer for Transformer model.
            tokenizer (Tokenizer): Tokenizer for processing expressions.
            config (dict): Configuration parameters.
        """
        self.lstm_trainer = lstm_trainer
        self.improved_lstm_trainer = improved_lstm_trainer
        self.transformer_trainer = transformer_trainer
        self.tokenizer = tokenizer
        self.config = config
        self.models_to_evaluate = ['lstm', 'improved_lstm', 'transformer']

    def get_trainer(self, model_name):
        """
        Get the appropriate trainer based on model name.
        
        Args:
            model_name (str): Name of the model.
            
        Returns:
            Trainer: The corresponding trainer instance.
        """
        if model_name == 'lstm':
            return self.lstm_trainer
        elif model_name == 'improved_lstm':
            return self.improved_lstm_trainer
        elif model_name == 'transformer':
            return self.transformer_trainer
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def compare_models(self, test_functions):
        """
        Compare models on test functions.
        
        Args:
            test_functions (list): List of function strings to test.
            
        Returns:
            tuple: Results and metrics for the comparison.
        """
        print("Comparing models...")

        # Generate True Expansions
        test_expansions = []
        for func in test_functions:
            try:
                x = sp.Symbol('x')
                true_func = sp.sympify(func)
                true_expansion = sp.series(
                    true_func,
                    x,
                    x0=self.config['data_generation']['x0'],
                    n=self.config['data_generation']['expansion_order'] + 1
                ).removeO()
                test_expansions.append(str(true_expansion))
            except Exception as e:
                print(f"Error processing function {func}: {e}")
                test_expansions.append('')

        results = []

        for func, true_expansion_str in tqdm(zip(test_functions, test_expansions),
                                         desc="Evaluating Models",
                                         total=len(test_functions)):
            try:
                # Dictionary to store predictions from each model
                predictions = {'function': func, 'true_expansion': true_expansion_str}
                
                # Get predictions from each model
                for model_name in self.models_to_evaluate:
                    trainer = self.get_trainer(model_name)
                    pred = trainer.predict(func)
                    predictions[f'{model_name}_prediction'] = pred
                
                results.append(predictions)
            except Exception as e:
                print(f"Error processing function {func}: {e}")
                continue

        # Save results
        results_path = os.path.join(self.config['paths']['results_dir'], 'model_comparison.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Comparison results saved to {results_path}")

        # Calculate accuracy metrics
        metrics = self._calculate_metrics(results)

        return results, metrics

    def _calculate_metrics(self, results):
        """
        Calculate comprehensive accuracy metrics for all models.
        
        Args:
            results (list): List of dictionaries with predicted expansions.
            
        Returns:
            dict: Dictionary with metrics for each model.
        """
        def normalize_expansion(exp):
            """Normalize expansion string for comparison."""
            return exp.replace(' ', '').replace('**', '^')

        def calculate_metrics(predictions, true_expansions):
            """Calculate detailed metrics."""
            exact_matches = 0
            partial_matches = 0
            token_accuracy = 0

            for pred, true_exp in zip(predictions, true_expansions):
                if not true_exp:
                    continue  # Skip examples where true expansion is missing
                    
                pred_norm = normalize_expansion(pred)
                true_norm = normalize_expansion(true_exp)

                # Exact match
                if pred_norm == true_norm:
                    exact_matches += 1

                # Partial match (Levenshtein distance)
                distance = Levenshtein.distance(pred_norm, true_norm)
                max_len = max(len(pred_norm), len(true_norm))
                similarity = 1 - (distance / max_len)

                if similarity > 0.7:  # 70% similarity threshold
                    partial_matches += 1

                # Token-level accuracy
                pred_tokens = pred.split()
                true_tokens = true_exp.split()
                matching_tokens = sum(p == t for p, t in zip(pred_tokens, true_tokens))
                token_acc = matching_tokens / max(len(pred_tokens), len(true_tokens))
                token_accuracy += token_acc

            # Filter out empty true expansions
            valid_examples = sum(1 for exp in true_expansions if exp)

            return {
                'exact_match_rate': exact_matches / valid_examples if valid_examples > 0 else 0,
                'partial_match_rate': partial_matches / valid_examples if valid_examples > 0 else 0,
                'token_accuracy': token_accuracy / valid_examples if valid_examples > 0 else 0,
                'exact_matches': exact_matches,
                'partial_matches': partial_matches,
                'total_examples': valid_examples
            }

        # Initialize metrics dictionary
        comprehensive_metrics = {}
        
        # Calculate metrics for each model
        for model_name in self.models_to_evaluate:
            comprehensive_metrics[model_name] = calculate_metrics(
                [result[f'{model_name}_prediction'] for result in results],
                [result['true_expansion'] for result in results]
            )

        # Save metrics
        metrics_path = os.path.join(self.config['paths']['results_dir'], 'comprehensive_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)

        # Print results
        for model_name in self.models_to_evaluate:
            print(f"\n{model_name.upper()} Model Metrics:")
            for key, value in comprehensive_metrics[model_name].items():
                print(f"{key}: {value}")

        # Plotting
        self._plot_detailed_comparison(comprehensive_metrics)

        return comprehensive_metrics

    def _plot_detailed_comparison(self, metrics):
        """
        Create a comprehensive visualization of model performance.
        
        Args:
            metrics (dict): Dictionary with metrics for each model.
        """
        plt.figure(figsize=(12, 8))

        # Metrics to plot
        metric_names = ['exact_match_rate', 'partial_match_rate', 'token_accuracy']
        model_values = {
            model_name: [metrics[model_name][m] for m in metric_names]
            for model_name in self.models_to_evaluate
        }

        x = np.arange(len(metric_names))
        width = 0.25  # width of the bars
        
        # Set colors for each model
        colors = {'lstm': 'blue', 'improved_lstm': 'green', 'transformer': 'orange'}
        
        # Plot bars for each model
        for i, (model_name, values) in enumerate(model_values.items()):
            offset = (i - 1) * width
            bars = plt.bar(x + offset, values, width, label=model_name.replace('_', ' ').title(), color=colors[model_name])
            
            # Add value labels
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                         f'{value:.2f}', ha='center', va='bottom')

        plt.xlabel('Performance Metrics', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14)
        plt.xticks(x, [m.replace('_', ' ').title() for m in metric_names])
        plt.legend()
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plot_path = os.path.join(self.config['paths']['results_dir'], 'detailed_model_comparison.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"Detailed comparison plot saved to {plot_path}")
