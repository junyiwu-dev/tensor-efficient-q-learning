# Tensor-Efficient Q-Learning (TEQL)

This repository contains the implementation and experimental code for the paper "Tensor-Efficient High-Dimensional Q-learning".

## Abstract

TEQL is a reinforcement learning framework that combines low-rank tensor decomposition with improved exploration strategies to achieve superior sample efficiency in high-dimensional state-action spaces. The method introduces: (1) a frequency-based penalty mechanism to prevent overfitting to frequently visited state-action pairs, and (2) an Error-Uncertainty Guided Exploration (EUGE) strategy that integrates approximation errors with upper confidence bounds.

## Requirements

### Software Dependencies
- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- SciPy >= 1.7.0
- Pandas >= 1.3.0
- TensorLy >= 0.7.0
- PyTorch >= 1.9.0
- Gymnasium (OpenAI Gym) >= 0.26.0
- Pathos >= 0.2.8

### Installation

```bash
# Clone the repository
git clone https://github.com/junyiwu-dev/tensor-efficient-q-learning.git
cd teql

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
teql/
├── src/
│   ├── algorithms/
│   │   ├── tlr_original.py      # Original Tensor Low-Rank (TLR) baseline
│   │   ├── teql.py              # TEQL implementation with frequency penalty and EUGE
│   │   └── q_learning.py        # Standard Q-learning for comparison
│   ├── environments/
│   │   ├── pendulum.py          # Modified Pendulum environment
│   │   └── cartpole.py          # Continuous CartPole environment
│   ├── experiments/
│   │   └── experiment_runner.py # Experiment orchestration
│   └── utils/
│       └── utils.py              # Discretizer and utility functions
├── parameters/               # JSON configuration files for experiments
├── results/                 # Output directory for experimental results
├── figures/                 # Generated plots and visualizations
├── visualization/           # Plotting and analysis scripts
│   ├── plot_tlr_vs_teql.py
│   ├── plot_ablation_study.py
│   ├── convergence_analysis.py
│   ├── performance_comparison.py
│   └── discretization_sensitivity.py
└── main.py                  # Main execution script
```

## Baseline Implementation Acknowledgment

This implementation builds upon the tensor low-rank reinforcement learning framework introduced in:

**Rozada, S., Paternain, S., & Marques, A. G. (2024). Tensor and matrix low-rank value-function approximation in reinforcement learning. IEEE Transactions on Signal Processing, 72, 1634-1649.**

- Original paper: [IEEE TSP](https://ieeexplore.ieee.org/document/10478283)
- Original code repository: [https://github.com/sergiorozada12/tensor-low-rank-rl](https://github.com/sergiorozada12/tensor-low-rank-rl)

### Components from the Original Implementation

The following components are adapted from Rozada et al.'s implementation:

1. **Baseline TLR Algorithm** (`src/algorithms/tlr_original.py`): The original Tensor Low-Rank learning algorithm with block coordinate descent
2. **Environment Modifications**: 
   - Modified Pendulum environment with bounded angles and custom reward function
   - Continuous action CartPole environment
3. **Discretization Framework** (`src/utils/utils.py`): State-action space discretization utilities
4. **Tensor Decomposition Structure**: CP decomposition framework for Q-function approximation

### Our Contributions

Building upon the baseline, TEQL introduces:

1. **Frequency-Based Penalty Mechanism**: Prevents overfitting to frequently visited state-action pairs (λ-regularization in objective function)
2. **Error-Uncertainty Guided Exploration (EUGE)**: Novel exploration strategy combining approximation errors with UCB
3. **Experimental Evaluation**: Ablation studies and sensitivity analyses demonstrating improvements

## Reproducing Results

### 1. Main Comparison (TEQL vs TLR)

Run the primary comparison experiment:

```bash
python main.py
```

This executes 100 iterations comparing TEQL with the original TLR baseline on both Pendulum and CartPole environments. Results are saved in `results_backup/`.

### 2. Ablation Study (Frequency Penalty)

Evaluate the impact of the frequency penalty mechanism:

```bash
python run_ablation_study.py
```

This compares TEQL with and without the frequency penalty term (λ = 0.01 vs λ = 0).

### 3. Discretization Sensitivity Analysis

Test robustness across different discretization granularities:

```bash
python run_sensitivity_analysis.py
```

Evaluates five discretization levels from very coarse (256 bins) to very fine (32,000 bins) for Pendulum, and (6,400 to 12,800,000 bins) for CartPole.

### 4. Generate Visualizations

After running experiments, generate all figures:

```bash
# Individual visualization scripts
python visualization/plot_tlr_vs_teql.py
python visualization/plot_ablation_study.py
python visualization/convergence_analysis.py
python visualization/performance_comparison.py
python visualization/discretization_sensitivity.py
```

Generated figures are saved in `figures/` directory as PDF files.

## Configuration

Experiment parameters are defined in JSON files within the `parameters/` directory. Key parameters include:

- `episodes`: Number of training episodes (default: 10,000)
- `max_steps`: Maximum steps per episode (default: 100)
- `epsilon`: Initial exploration rate
- `alpha`: Learning rate (default: 0.005)
- `gamma`: Discount factor (default: 0.9)
- `k`: Tensor decomposition rank (default: 10)
- `lambda_penalty`: Frequency penalty weight (default: 0.01)
- `c`: Exploration constant for EUGE (default: 2.0)

### Example Configuration (pendulum_convergent_tlr_learning.json):

```json
{
    "type": "convergent-tlr",
    "bucket_states": [20, 20],
    "bucket_actions": [10],
    "episodes": 10000,
    "max_steps": 100,
    "epsilon": 1.0,
    "alpha": 0.005,
    "gamma": 0.9,
    "k": 10,
    "lambda_penalty": 0.01,
    "c": 1.0
}
```

## Key Algorithm Components

### 1. Low-Rank Tensor Q-Function Update

The Q-function is approximated as a rank-R tensor using CP decomposition:
- Located in `src/algorithms/teql.py`, method `update_q_matrix()`
- Implements block coordinate descent with frequency-based penalty

### 2. Error-Uncertainty Guided Exploration (EUGE)

Action selection mechanism combining approximation error with UCB:
- Located in `src/algorithms/teql.py`, method `choose_action()`
- Computes EU value: EU_t(s,a) = Q̂(s,a) + c·[Q_error(s,a) + √(log N_total/N(s,a))]

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please contact:
- Junyi Wu (junyiwu@uw.edu)

Department of Industrial & System Engineering, University of Washington
