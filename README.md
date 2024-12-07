# Tensor and Matrix Low-Rank Value-Function Approximation in Reinforcement Learning

This project implements novel approaches for value function approximation in reinforcement learning using low-rank matrix and tensor decomposition techniques. It provides efficient algorithms that reduce the computational complexity and memory requirements while maintaining performance.

## Features

- Matrix Low-Rank (MLR) learning algorithm for value function approximation
- Tensor Low-Rank (TLR) learning algorithm for handling high-dimensional state-action spaces
- Count-based exploration strategy integrated with MLR (CountMLR)
- Implementations of standard RL environments (pendulum, cartpole, mountain car, etc.)
- Comprehensive experimental framework for comparing different methods
- Visualization tools for analyzing results

## Project Structure

- `src/algorithms/` - Core algorithm implementations (MLR, TLR, Q-learning, etc.)
- `src/environments/` - Custom RL environment implementations
- `src/experiments/` - Experiment runners and configuration
- `src/utils/` - Utility functions and helper classes
- `parameters/` - JSON configuration files for different experiments
- `results/` - Directory for storing experimental results
- `figures/` - Output directory for generated plots and visualizations

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Gym
- TensorLy
- Pathos

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Experiments

The project includes several experiment scripts that can be run directly:

```bash
# Run matrix low-rank comparison experiments
python 5_exp_mlr_comparison.py

# Run methods comparison experiments
python 5_exp_methods_comparison.py

# Run high-dimensional experiments
python 7_exp_high_dimensional_space.py
```

### Configuration

Experiments can be configured through JSON files in the `parameters/` directory. Example configurations are provided for different environments:

```json
{
    "type": "mlr-model",
    "min_points_states": [-1, -5],
    "max_points_states": [1, 5],
    "bucket_states": [20, 20],
    "min_points_actions": [-2],
    "max_points_actions": [2],
    "bucket_actions": [10],
    "episodes": 15000,
    "max_steps": 100,
    "epsilon": 1.0,
    "alpha": 0.01,
    "gamma": 0.9,
    "decay": 0.999999,
    "k": 4
}
```

## Supported Environments

- Pendulum
- Cartpole
- Mountain Car
- Goddard Rocket
- Highway
- Wireless Communications

## Experimental Results

The project includes comprehensive experiments comparing:
- Traditional Q-learning vs MLR/TLR approaches
- Performance across different environment complexities
- Scalability analysis
- Computational complexity comparisons
- High-dimensional problem handling

Results are automatically saved to the `results/` directory and visualizations to `figures/`.

## Citations

@article{rozada2024tensor,
  title={Tensor and Matrix Low-Rank Value-Function Approximation in Reinforcement Learning},
  author={Rozada, Sergio and Paternain, Santiago and Marques, Antonio G},
  journal={IEEE Transactions on Signal Processing},
  volume={72},
  pages={1634--1649},
  year={2024},
  publisher={IEEE}
}
```
