import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# Set style for academic publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# Create synthetic data for demonstration
np.random.seed(42)
grid_size = 20

# Create state-action space grid
state_dim = np.linspace(-2, 2, grid_size)
action_dim = np.linspace(-1, 1, grid_size)
S, A = np.meshgrid(state_dim, action_dim)

# Simulate Q-error distribution (approximation uncertainty)
# Higher errors in complex regions (corners and specific patterns)
q_error = np.zeros((grid_size, grid_size))

# Create regions with different error patterns
# Region 1: High error in top-right corner (complex Q-function structure)
for i in range(grid_size):
    for j in range(grid_size):
        # Base error from distance to origin
        dist_from_center = np.sqrt((i - grid_size/2)**2 + (j - grid_size/2)**2)
        
        # Complex regions with high approximation error
        if i > 14 and j > 14:  # Top-right corner
            q_error[i, j] = 0.8 + 0.2 * np.random.rand()
        elif i < 5 and j < 5:  # Bottom-left corner
            q_error[i, j] = 0.7 + 0.2 * np.random.rand()
        elif 8 < i < 12 and 8 < j < 12:  # Center region (well-explored)
            q_error[i, j] = 0.1 + 0.1 * np.random.rand()
        else:
            # Smooth transition areas
            q_error[i, j] = 0.3 + 0.2 * np.sin(i/3) * np.cos(j/3) + 0.1 * np.random.rand()

# Apply smoothing for realistic appearance
from scipy.ndimage import gaussian_filter
q_error = gaussian_filter(q_error, sigma=1.0)

# Simulate visit counts (inverse relationship with uncertainty in some areas)
visit_counts = np.zeros((grid_size, grid_size))
for i in range(grid_size):
    for j in range(grid_size):
        if 8 < i < 12 and 8 < j < 12:  # Center well-explored
            visit_counts[i, j] = 50 + 30 * np.random.rand()
        else:
            visit_counts[i, j] = 5 + 10 * np.random.rand()

# Calculate EUGE scores (exploration priority)
# EU_t(s,a) = Q(s,a) + c * (Q_error(s,a) + sqrt(log(N_total)/N(s,a)))
c = 0.5  # exploration constant
N_total = np.sum(visit_counts)
ucb_bonus = np.sqrt(np.log(N_total + 1) / (visit_counts + 1))
euge_scores = q_error + c * ucb_bonus

# Normalize for visualization
euge_scores = (euge_scores - euge_scores.min()) / (euge_scores.max() - euge_scores.min())

# Create figure with subplots
fig = plt.figure(figsize=(15, 5))

# Define custom colormap for better visualization
colors_error = ['#2E7D32', '#66BB6A', '#FFF59D', '#FFB74D', '#E65100']
n_bins = 100
cmap_error = LinearSegmentedColormap.from_list('q_error', colors_error, N=n_bins)

colors_euge = ['#1A237E', '#3949AB', '#7986CB', '#FFCA28', '#FF6F00']
cmap_euge = LinearSegmentedColormap.from_list('euge', colors_euge, N=n_bins)

# Subplot 1: Q-error distribution
ax1 = fig.add_subplot(131)
im1 = ax1.imshow(q_error.T, extent=[-2, 2, -1, 1], origin='lower', 
                 cmap=cmap_error, aspect='auto', alpha=0.9)
ax1.set_xlabel('State Dimension', fontweight='bold')
ax1.set_ylabel('Action Dimension', fontweight='bold')
ax1.set_title('(a) Q-Error Distribution\n(Approximation Uncertainty)', fontweight='bold')
ax1.grid(True, alpha=0.2, linestyle='--')

# Removed contour lines for cleaner visualization

# Colorbar for Q-error
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar1.set_label('Q-Error Magnitude', rotation=270, labelpad=20)

# Subplot 2: Visit counts (for context)
ax2 = fig.add_subplot(132)
im2 = ax2.imshow(visit_counts.T, extent=[-2, 2, -1, 1], origin='lower', 
                 cmap='YlOrBr', aspect='auto', alpha=0.9)
ax2.set_xlabel('State Dimension', fontweight='bold')
ax2.set_ylabel('Action Dimension', fontweight='bold')
ax2.set_title('(b) Visit Frequency\n(Historical Exploration)', fontweight='bold')
ax2.grid(True, alpha=0.2, linestyle='--')

# Colorbar for visit counts
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Visit Count', rotation=270, labelpad=20)

# Subplot 3: EUGE scores with exploration priority arrows
ax3 = fig.add_subplot(133)
im3 = ax3.imshow(euge_scores.T, extent=[-2, 2, -1, 1], origin='lower', 
                 cmap=cmap_euge, aspect='auto', alpha=0.9)
ax3.set_xlabel('State Dimension', fontweight='bold')
ax3.set_ylabel('Action Dimension', fontweight='bold')
ax3.set_title('(c) EUGE Priority Score\n(Exploration Guidance)', fontweight='bold')
ax3.grid(True, alpha=0.2, linestyle='--')

# Removed arrows for cleaner visualization

# Add text annotations for key regions with higher transparency
ax3.text(1.3, 0.7, 'High\nPriority', fontsize=10, color='red', 
         fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor="white", alpha=0.3))
ax3.text(-1.3, -0.7, 'High\nPriority', fontsize=10, color='red', 
         fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor="white", alpha=0.3))
ax3.text(0, 0, 'Well\nExplored', fontsize=10, color='blue', 
         fontweight='bold', ha='center', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor="white", alpha=0.3))

# Colorbar for EUGE scores
cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar3.set_label('EUGE Score', rotation=270, labelpad=20)

# Removed overall title and equation
plt.tight_layout()

# Save figure with high DPI for poster quality
plt.savefig('EUGE_visualization_poster.pdf', dpi=300, bbox_inches='tight', 
            format='pdf', metadata={'Title': 'EUGE Visualization',
                                   'Author': 'TEQL Research Team',
                                   'Subject': 'Error-Uncertainty Guided Exploration'})

# Also save as PNG for preview
plt.savefig('EUGE_visualization_poster.png', dpi=300, bbox_inches='tight')

plt.show()

print("Visualization saved as 'EUGE_visualization_poster.pdf' with 300 DPI")
print("Key features visualized:")
print("1. Q-error distribution showing approximation uncertainty")
print("2. Visit frequency showing historical exploration patterns") 
print("3. EUGE scores with arrows indicating exploration priority")
print("4. Clear identification of under-explored high-uncertainty regions")