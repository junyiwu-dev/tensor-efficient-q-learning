# Tensor-Efficient Q-Learning (TEQL)

## Supplementary Code for Technometrics Submission

**Paper Title:** Tensor-Efficient High-Dimensional Q-Learning

This repository contains the complete implementation and reproducible code for all experimental results presented in the paper.

---

## Quick Start: Reproducing Paper Results

### Minimal Reproduction (Recommended for Review)

The following commands reproduce key figures from the paper in approximately **2-3 hours** on a standard laptop:

```bash
# Step 1: Install dependencies
pip install numpy torch tensorly gymnasium matplotlib scipy pandas seaborn

# Step 2: Run comparison experiment (5 iterations for quick verification)
python main_comparison.py --env cartpole pendulum --model teql tlr --iterations 5

# Step 3: Generate Figure 2 (Learning Curves)
python plot_baselines.py --env cartpole pendulum --model teql tlr
```

**Output:** `figures/comparison_all_algorithms.pdf` corresponds to **Figure 2** in the paper.

---

## Complete Reproduction Guide

### System Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.8 or higher |
| RAM | 8 GB minimum |
| Storage | 2 GB for results |
| OS | Linux, macOS, or Windows |

### Software Dependencies

```
numpy>=1.20.0
torch>=1.9.0
tensorly>=0.7.0
gymnasium>=0.26.0
matplotlib>=3.3.0
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.0
highway-env>=1.8.0  # Optional: for Highway environment
```

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy torch tensorly gymnasium matplotlib scipy pandas seaborn

# Optional: Install highway-env for Highway experiments
pip install highway-env
```

---

## Reproducing Specific Figures and Tables

### Figure 2: Main Algorithm Comparison (TEQL vs Baselines)

**Description:** Learning curves comparing TEQL, TLR, LoRa-VI, DQN, and SAC on CartPole and Pendulum.

**Commands:**
```bash
# Full reproduction (30 iterations, ~10 hours)
python main_comparison.py --env cartpole pendulum --model teql tlr lora-vi dqn sac --iterations 30

# Generate plot
python plot_baselines.py --env cartpole pendulum --model teql tlr lora-vi dqn sac
```

**Output:** `figures/comparison_all_algorithms.pdf`

**Estimated Time:** 
- Quick version (5 iterations): ~2 hours
- Full version (30 iterations): ~10 hours

---

### Figure 3: Ablation Study (Effect of λ Regularization)

**Description:** Comparison of TEQL with λ=0 versus λ>0 demonstrating the impact of frequency-based penalty.

**Commands:**
```bash
# Run ablation experiments
python main_comparison.py --env cartpole pendulum --model teql --iterations 30

# Generate ablation plots
python ablation.py
```

**Output:** 
- `figures/ablation_curves.pdf` - Learning curves
- `figures/ablation_boxplot.pdf` - Final performance distribution

**Estimated Time:** ~4 hours

---

### Figure 4: Discretization Sensitivity Analysis

**Description:** Performance across different discretization granularities (very_coarse to very_fine).

**Commands:**
```bash
# Run TEQL sensitivity analysis
python sensitivity_analysis.py --env cartpole pendulum --iterations 30

# Run TLR sensitivity analysis (for comparison)
python sensitivity_analysis_TLR.py --env cartpole pendulum --iterations 30

# Generate sensitivity plots
python plot_sensitivity.py --env cartpole pendulum
python plot_sensitivity_tlr_vs_teql.py
```

**Output:**
- `figures/sensitivity_analysis.pdf` - TEQL across discretization levels
- `figures/sensitivity_tlr_vs_teql.pdf` - TLR vs TEQL comparison

**Estimated Time:** ~8 hours

---

### Table 1: Final Performance Statistics

**Description:** Mean and standard deviation of final cumulative rewards.

**Commands:**
```bash
# Generate comprehensive analysis with statistics
python comprehensive_analysis_plots.py
```

**Output:** 
- `figures/final_performance_statistics.txt` - Numerical results
- Console output with formatted statistics table

---

## File Descriptions

### Core Algorithms

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `teql.py` | **TEQL** - Proposed method with frequency penalty and EUGE exploration | Section 3 |
| `tlr_original.py` | **TLR** - Tensor Low-Rank baseline (Rozada et al., 2024) | Section 5.1 |
| `lora_vi.py` | **LoRa-VI** - Low-Rank Value Iteration baseline | Section 5.1 |
| `dqn.py` | **DQN** - Deep Q-Network baseline | Section 5.1 |
| `sac.py` | **SAC** - Soft Actor-Critic baseline | Section 5.1 |
| `q_learning.py` | Standard tabular Q-Learning | Appendix |

### Environments

| File | Description |
|------|-------------|
| `cartpole.py` | Continuous CartPole with custom reward function |
| `pendulum.py` | Modified Pendulum with bounded angles |
| `highway_wrapper.py` | Highway driving environment wrapper |
| `mountaincar.py` | Continuous Mountain Car environment |

### Experiment Scripts

| File | Description | Reproduces |
|------|-------------|------------|
| `main_comparison.py` | Main experiment runner for algorithm comparison | Figure 2 |
| `ablation.py` | Ablation study for λ regularization | Figure 3 |
| `sensitivity_analysis.py` | TEQL discretization sensitivity | Figure 4 |
| `sensitivity_analysis_TLR.py` | TLR discretization sensitivity | Figure 4 |
| `comprehensive_analysis_plots.py` | Statistical analysis and box plots | Table 1 |

### Visualization Scripts

| File | Description |
|------|-------------|
| `plot_baselines.py` | Multi-algorithm comparison plots |
| `plot_sensitivity.py` | Sensitivity analysis visualization |
| `plot_sensitivity_tlr_vs_teql.py` | TLR vs TEQL comparison plots |
| `plot_two_env.py` | Two-environment comparison layout |

---

## Algorithm Parameters

Key hyperparameters used in experiments (see paper Section 5.2):

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Episodes | T | 10,000 | Number of training episodes |
| Max steps | H | 100 | Maximum steps per episode |
| Learning rate | α | 0.005 | Step size for updates |
| Discount factor | γ | 0.9 | Future reward discount |
| Tensor rank | k | 10 | CP decomposition rank |
| Frequency penalty | λ | 1e-5 | Regularization weight |
| UCB constant | c | 1.0 | Exploration-exploitation balance |
| Initial ε | ε₀ | 1.0 | Initial exploration rate |

---

## Expected Output Structure

After running experiments, results are organized as:

```
├── results_backup/
│   ├── iteration_0/
│   │   ├── cartpole_convergent_tlr_learning.json  # TEQL results
│   │   ├── cartpole_original_tlr_learning.json    # TLR results
│   │   ├── cartpole_dqn.json                      # DQN results
│   │   └── ...
│   ├── iteration_1/
│   └── ...
├── figures/
│   ├── comparison_all_algorithms.pdf              # Figure 2
│   ├── ablation_curves.pdf                        # Figure 3a
│   ├── ablation_boxplot.pdf                       # Figure 3b
│   ├── sensitivity_analysis.pdf                   # Figure 4a
│   └── sensitivity_tlr_vs_teql.pdf                # Figure 4b
```

---

## Verification Checklist

To verify successful reproduction:

- [ ] `figures/comparison_all_algorithms.pdf` shows TEQL outperforming baselines
- [ ] TEQL achieves higher final rewards than TLR on both environments
- [ ] Ablation study shows λ>0 improves stability over λ=0
- [ ] Sensitivity analysis shows robustness across discretization levels

---

## Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'highway_env'`
**Solution:** Highway environment is optional. Install with `pip install highway-env` or skip highway experiments.

**Issue:** Slow execution
**Solution:** Reduce iterations: `--iterations 5` for quick verification.

**Issue:** Memory error
**Solution:** Run environments separately: `--env cartpole` then `--env pendulum`.

---

## Baseline Acknowledgment

This implementation builds upon the tensor low-rank RL framework from:

> Rozada, S., Paternain, S., & Marques, A. G. (2024). Tensor and matrix low-rank value-function approximation in reinforcement learning. *IEEE Transactions on Signal Processing*, 72, 1634-1649.

---

## License

MIT License

---

## Contact

For questions regarding code reproduction, please contact the corresponding author.
