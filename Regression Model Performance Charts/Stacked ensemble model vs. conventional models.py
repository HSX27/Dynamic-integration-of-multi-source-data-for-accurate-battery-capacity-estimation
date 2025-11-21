import matplotlib.pyplot as plt
import numpy as np

# Data preparation:
conditions = ['RF', 'Ridge', 'NN', 'XGB', 'LGBM', 'MLP', 'KNN', 'SVM']
r2   = [0.0860, 0.4617, 1.3064, 0.0606, 0.0943, 0.1651, 0.0718, 0.2113]
mae  = [0.5865, 0.7556, 0.8063, 0.5217, 0.5769, 0.6043, 0.2361, 0.7660]
rmse = [0.5946, 0.7826, 0.8358, 0.5385, 0.6087, 0.6853, 0.5652, 0.7134]

# Convert to percentage
r2_percent = [x * 100 for x in r2]
mae_percent = [x * 100 for x in mae]
rmse_percent = [x * 100 for x in rmse]

# Plot parameter settings
bar_width = 0.25
index = np.arange(len(conditions))
colors = {
    'R2': '#f38181',
    'MAE': '#fce38a',
    'RMSE': '#95e1d3'
}

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot R² bar chart
bar1 = ax1.bar(index - bar_width, r2_percent, bar_width,
               color=colors['R2'], edgecolor='black', linewidth=1.5, label='R²')
ax1.set_xlabel('Model', fontsize=16, fontweight='bold')
ax1.set_ylabel('R² (%)', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax1.set_ylim(0, 150)
ax1.set_xticks(index)
ax1.set_xticklabels(conditions, fontsize=14, fontweight='bold', rotation=0)

# Create MAE axis
ax2 = ax1.twinx()
bar2 = ax2.bar(index, mae_percent, bar_width,
               color=colors['MAE'], edgecolor='black', linewidth=1.5, label='MAE')
ax2.set_ylabel('MAE (%)', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax2.set_ylim(0, 100)

# Create RMSE axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
bar3 = ax3.bar(index + bar_width, rmse_percent, bar_width,
               color=colors['RMSE'], edgecolor='black', linewidth=1.5, label='RMSE')
ax3.set_ylabel('RMSE (%)', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax3.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax3.set_ylim(0, 100)

# Unify axis styles
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(True)
    for side in ['left', 'right', 'bottom', 'top']:
        ax.spines[side].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=14)

# Adjust legend position and style
ax1.legend([bar1[0], bar2[0], bar3[0]], ['R²', 'MAE', 'RMSE'],
           loc='upper left', fontsize=12, frameon=False,
           bbox_to_anchor=(0.83, 0.98),
           handletextpad=0.5, columnspacing=1.2)

# Save as high-resolution PDF and display
plt.tight_layout(pad=3)
plt.show()