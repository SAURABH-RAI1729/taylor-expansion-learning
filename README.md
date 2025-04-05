# Taylor Expansion Learning

This project implements neural network models to learn Taylor series expansions of mathematical functions. By training sequence-to-sequence models to predict Taylor expansions, we explore how different architectures can capture mathematical patterns and symbolic relationships.

## Project Overview

The Taylor Expansion Learning project demonstrates the application of deep learning to symbolic mathematics. Taylor series are fundamental in mathematical analysis, and automating their derivation showcases neural networks' capabilities in understanding mathematical transformations.

## Key Features

- **Dataset Generation**: Programmatic creation of diverse mathematical functions and their corresponding Taylor expansions using SymPy
- **Multiple Model Architectures**:
  - Basic LSTM Sequence-to-Sequence Model
  - Improved LSTM with Bidirectional Encoder and Attention Mechanism
  - Transformer-based Model
- **Comprehensive Evaluation Framework**: Compare models using exact match rate, partial match rate (>= 70%), and token accuracy metrics
- **Interactive Demo**: Test models with custom mathematical functions

## Installation

```bash
# Clone the repository
git clone https://github.com/SAURABH-RAI1729/taylor-expansion-learning.git
cd taylor-expansion-learning

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

For a complete demonstration (scaled-down version) of all components:

```bash
python scripts/demo.py
```

To train and evaluate models individually:

```bash
# Train LSTM model
python scripts/train_lstm.py

# Train Improved LSTM model
python scripts/train_improved_lstm.py

# Train Transformer model
python scripts/train_transformer.py

# Compare all models
python scripts/compare_models.py
```

## Data Generation and Preprocessing

### Data Generation

The system generates mathematical functions and their Taylor expansions using the SymPy library. It creates a diverse dataset by:

1. Selecting from basic functions (sin, cos, exp, log, etc.)
2. Applying transformations (scaling, shifting)
3. Creating compositions with polynomials
4. Computing Taylor expansions around a specified point (default: x=0)

This simple approach ensures a rich dataset that covers various types of mathematical functions with different behaviors.

### Tokenization Strategy

The tokenization approach is critical for processing mathematical expressions. Our implementation:

- **Mathematical Token Recognition**: Special handling for mathematical operators, functions, and symbols
- **Function-Specific Tokenization**: Separate tokenization for functions like `sin`, `cos`, `exp`, etc.
- **Operator Handling**: Proper tokenization of operators (`+`, `-`, `*`, `/`, `^`, `**`)
- **Special Tokens**: Includes `<PAD>`, `<UNK>`, `<SOS>` (start of sequence), and `<EOS>` (end of sequence) tokens
- **Vocabulary Building**: Dynamic vocabulary construction from training data

Example of tokenization:
```
"sin(2*x) + x^2" → ["<SOS>", "sin", "(", "2", "*", "x", ")", "+", "x", "^", "2", "<EOS>"]
```

This simple approach maintains the mathematical structure while creating a format suitable for sequence models.

## Model Architectures

### LSTM Sequence-to-Sequence Model

The basic LSTM model follows a standard encoder-decoder architecture:

- **Encoder**: LSTM network that processes the input function and produces a hidden state representation
- **Decoder**: LSTM network that generates the Taylor expansion token by token
- **Teacher Forcing**: Uses ground truth tokens during training for better convergence
- **Architecture Details**:
  - Embedding dimension: 128
  - Hidden dimension: 256
  - Layers: 2
  - Dropout: 0.2

This baseline model provides a foundation for comparison with more advanced architectures.

### Improved LSTM with Attention

The improved LSTM model enhances the basic architecture with:

- **Bidirectional Encoder**: Processes input sequences in both directions to capture broader context
- **Attention Mechanism**: Allows the decoder to focus on different parts of the input function at each decoding step
- **Enhanced Architecture**:
  - Embedding dimension: 256 (increased from baseline)
  - Hidden dimension: 512 (increased from baseline)
  - Layers: 3 (deeper network)
  - Attention dimension: 256
  - Dropout: 0.3 (slightly higher for regularization)

The attention mechanism calculates relevance scores between decoder states and encoder outputs, allowing the model to "focus" on important parts of the input while generating each output token.

Mathematically, the attention mechanism works by:
1. Calculating energy scores between the current decoder state and all encoder states
2. Normalizing these scores with softmax to get attention weights
3. Creating a context vector as a weighted sum of encoder outputs
4. Concatenating the context vector with the current decoder input for prediction

This architecture significantly improves the model's ability to handle complex mathematical patterns and longer sequences.

### Transformer Model

The Transformer model implements the architecture from "Attention Is All You Need" (Vaswani et al.):

- **Self-Attention Mechanism**: Enables parallel processing and global context awareness
- **Multi-head Attention**: Allows the model to jointly attend to information from different representation subspaces
- **Positional Encoding**: Injects information about token positions since Transformers have no inherent positional awareness
- **Architecture Details**:
  - Embedding dimension: 128
  - Number of attention heads: 8
  - Encoder layers: 4
  - Decoder layers: 4
  - Feedforward dimension: 1024
  - Dropout: 0.2


## Performance Evaluation

Models are evaluated on three key metrics:

1. **Exact Match Rate**: Percentage of predictions that perfectly match the ground truth
2. **Partial Match Rate**: Percentage of predictions with high similarity to the ground truth (Levenshtein distance, 0.7)
3. **Token Accuracy**: Average percentage of correctly predicted tokens

Sample Results:

| Model | Exact Match Rate | Partial Match Rate | Token Accuracy |
|-------|------------------|-------------------|----------------|
| LSTM | 32.9% | 72.4% | 0.6% |
| Improved LSTM | 65.7% | 91.8% | 1.3% |
| Transformer | 52.9% | 83.2% | 3.7% |

These are just preliminary results and may change on hyperparameter fine-tuning.

## Project Structure

```
taylor_expansion_learning/
├── README.md - Project documentation
├── requirements.txt - Dependencies
├── setup.py - Package setup
├── taylor_expansion_learning/ - Main package
│   ├── __init__.py
│   ├── config.py - Configuration parameters
│   ├── data/ - Data handling components
│   │   ├── __init__.py
│   │   ├── generator.py - Taylor expansion data generation
│   │   ├── processor.py - Data processing and batching
│   │   └── tokenizer.py - Mathematical expression tokenization
│   ├── models/ - Model architectures
│   │   ├── __init__.py
│   │   ├── lstm_model.py - Basic LSTM model
│   │   ├── improved_lstm_model.py - LSTM with attention
│   │   └── transformer_model.py - Transformer model
│   ├── training/ - Training utilities
│   │   ├── __init__.py
│   │   ├── lstm_trainer.py - LSTM trainer
│   │   ├── improved_lstm_trainer.py - Improved LSTM trainer
│   │   └── transformer_trainer.py - Transformer trainer
│   └── evaluation/ - Evaluation utilities
│       ├── __init__.py
│       └── evaluator.py - Model evaluation and comparison
└── scripts/ - Training and evaluation scripts
    ├── train_lstm.py - Train LSTM model
    ├── train_improved_lstm.py - Train improved LSTM model
    ├── train_transformer.py - Train transformer model
    ├── compare_models.py - Compare all models
    └── demo.py - End-to-end demonstration
    └── Results.ipynb - Jupyter notebook showing results
    
```

## Future Directions

1. **Model Improvements**:
   - Explore hybrid architectures combining LSTM and Transformer components
   - Explore other attention mechanisms

2. **Dataset Enhancements**:
   - Expand to higher-order expansions
   - Include more complex mathematical functions
   - Add multi-variable Taylor expansions

## Conclusion

This project demonstrates the application of modern deep learning architectures to symbolic mathematics. The significant performance improvement of the attention-based models highlights the importance of capturing global context in mathematical expressions.

## License

MIT

## Acknowledgments

This project has been developed as a submission for Google Summer of Code (GSoC) 2025.
