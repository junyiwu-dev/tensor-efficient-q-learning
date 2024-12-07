import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from src.environments.mountaincar import CustomContinuous_MountainCarEnv
from src.experiments.experiments import Experiment
from src.utils.utils import OOMFormatter

# Initialize environment
env_mountaincar = CustomContinuous_MountainCarEnv()

N_NODES = 1

if __name__ == "__main__":
    # Mountain Car
    name = "mountaincar_mlr_learning.json"
    experiment = Experiment(name, env_mountaincar, N_NODES)
    experiment.run_experiments(window=50)

    # Plot results
    plt.style.use('seaborn-v0_8')
    matplotlib.rcParams.update({
        'font.size': 18,
        'figure.figsize': (8, 6),
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    fig, ax = plt.subplots()
    
    # Plot steps
    prefix = 'results/'
    steps = range(0, 5000, 10)
    
    # Mountain Car
    ax.plot(steps, json.load(open(prefix + 'mountaincar_mlr_learning.json', 'r'))['steps'], color='g', label='MLR-learning')
    ax.set_xlabel("Episodes", labelpad=4)
    ax.set_ylabel("Mountain Car # Steps")
    ax.set_yticks([0, 5000, 10000])
    ax.set_xticks([0, 2500, 5000])
    ax.yaxis.set_major_formatter(OOMFormatter(4, "%1.1f"))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_xlim(0, 5000)
    ax.legend(fontsize=12)

    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/mountaincar_mlr_learning.jpg', dpi=300, bbox_inches='tight')
    
# import os
# import json
# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np

# from src.environments.pendulum import CustomPendulumEnv
# from src.environments.cartpole import CustomContinuousCartPoleEnv
# from src.environments.mountaincar import CustomContinuous_MountainCarEnv
# from src.environments.goddard import CustomGoddardEnv

# from src.experiments.experiments import Experiment
# from src.experiments.experiment_MLR import MLRExperiment
# from src.utils.utils import OOMFormatter

# env_pendulum = CustomPendulumEnv()
# env_cartpole = CustomContinuousCartPoleEnv()
# env_mountaincar = CustomContinuous_MountainCarEnv()
# env_rocket = CustomGoddardEnv()

# N_NODES = 1

# if __name__ == "__main__":
#     # Pendulum
#     name = "pendulum_mlr_learning.json"
#     experiment = MLRExperiment(name, env_pendulum, N_NODES)
#     experiment.run_experiments(window=30)

#     # Cartpole
#     name = "cartpole_mlr_learning.json"
#     experiment = MLRExperiment(name, env_cartpole, N_NODES)
#     experiment.run_experiments(window=50)

#     # Plot results
#     plt.style.use('seaborn-v0_8')
#     matplotlib.rcParams.update({
#         'font.size': 18,
#         'figure.figsize': (12, 5),
#         'lines.linewidth': 2,
#         'axes.grid': True,
#         'grid.alpha': 0.3
#     })
    
#     fig, (ax1, ax2) = plt.subplots(1, 2)
    
#     # Plot steps
#     prefix = 'results/'
#     steps_pendulum = range(0, 15000, 10)
#     steps_cartpole = range(0, 40000, 10)
    
#     # Pendulum
#     ax1.plot(steps_pendulum, json.load(open(prefix + 'pendulum_mlr_learning.json', 'r'))['steps'], color='g', label='MLR-learning')
#     ax1.set_xlabel("Episodes", labelpad=4)
#     ax1.set_ylabel("Pendulum # Steps")
#     ax1.set_xlim(0, 15000)
#     ax1.set_yticks([0, 50, 100])
#     ax1.set_xticks([0, 7500, 15000])
#     ax1.yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
#     ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax1.legend(fontsize=12)

#     # Cartpole
#     ax2.plot(steps_cartpole, json.load(open(prefix + 'cartpole_mlr_learning.json', 'r'))['steps'], color='g', label='MLR-learning')
#     ax2.set_xlabel("Episodes", labelpad=4)
#     ax2.set_ylabel("Cartpole # Steps")
#     ax2.set_yticks([0, 50, 100])
#     ax2.set_xticks([0, 20000, 40000])
#     ax2.yaxis.set_major_formatter(OOMFormatter(2, "%1.1f"))
#     ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     ax2.set_xlim(0, 40000)
#     ax2.legend(fontsize=12)

#     plt.tight_layout()
    
#     # 确保figures目录存在
#     os.makedirs('figures', exist_ok=True)
#     plt.savefig('figures/mlr_learning_comparison.jpg', dpi=300, bbox_inches='tight')