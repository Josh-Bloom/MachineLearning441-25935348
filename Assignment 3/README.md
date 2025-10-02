# Incremental Learning with Dynamic Neural Networks

This project implements incremental learning using dynamic neural networks that can grow their architecture during training. The networks learn new classes over time without forgetting previously learned ones.

## Key Features

- Dynamic architecture growth (add hidden neurons and output classes)
- Weight preservation when adding new components
- Automatic architecture search based on overfitting detection
- Cross-validation evaluation with statistical significance testing

## Project Structure

```
Assignment 3/
├── README.md             # This file
├── core.py               # Main training and evaluation functions
├── dynamicNN.py          # Dynamic neural network implementation
├── data.py               # Data loading and preprocessing
├── config.py             # Configuration management
├── model_util.py         # Utility functions
├── evalu.py              # Statistical analysis and visualisation
└── datasets/             # Cached dataset files
```

## Installation

Developed and tested with Python 3.11.7.

```bash
pip install torch numpy scikit-learn pandas matplotlib seaborn ucimlrepo imbalanced-learn scipy
```

## Usage

`core.py` has both a main function and a module function for running the experiments. **Warning**: execution for 5 seeds, 5 datasets, and 4-fold CV can take up to 4.5 hours.

### Available Datasets

- **glass**: Glass identification (6 classes)
- **sil**: Vehicle silhouettes (4 classes)  
- **segment**: Image segmentation (7 classes)
- **wine**: Wine quality (6 classes)
- **yeast**: Yeast protein localisation (10 classes)

## Configuration

`config.py` contains all training, cross-validation, architecture, and so on parameters.

## How It Works

1. Start with 2 smallest classes
2. Add hidden neurons until overfitting detected
3. Add remaining classes incrementally
4. Preserve weights when adding new components

See `core.py` for the main training logic and `dynamicNN.py` for the network implementation.

## Results

The system outputs:
- Accuracy and F1-score comparisons
- Hidden neuron count analysis
- Statistical significance testing (Wilcoxon test)
- Visualisation plots (`accuracy.pdf`, `neurons.pdf`)

See `evalu.py` for statistical analysis functions.