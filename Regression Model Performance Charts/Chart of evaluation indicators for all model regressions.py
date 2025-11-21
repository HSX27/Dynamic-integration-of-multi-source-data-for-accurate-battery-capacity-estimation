import matplotlib.pyplot as plt
import numpy as np

# Data preparation:
conditions = ['SE', 'RF', 'Ridge', 'NN', 'XGB', 'LGBM', 'MLP', 'KNN', 'SVM']
r2   = [0.9839, 0.9060, 0.6731, 0.4266, 0.9277, 0.8991, 0.8445, 0.9180, 0.8123]
mae  = [0.0058, 0.0133, 0.0225, 0.0284, 0.0115, 0.0130, 0.0139, 0.0072, 0.0235]
rmse = [0.0092, 0.0222, 0.0414, 0.0548, 0.0195, 0.0230, 0.0286, 0.0207, 0.0314]

# Plot parameter settings
bar_width = 0.25
index = np.arange(len(conditions))
colors = {
    'R2': '#9DC7DD',
    'MAE': '#30A5C2',
    'RMSE': '#EEF8B4'
}

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot R² bar chart
bar1 = ax1.bar(index - bar_width, r2, bar_width,
               color=colors['R2'], edgecolor='black', linewidth=1.5, label='R²')
ax1.set_xlabel('Model', fontsize=16, fontweight='bold')
ax1.set_ylabel('R² Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax1.set_ylim(0.4, 1.0)
ax1.set_xticks(index)
ax1.set_xticklabels(conditions, fontsize=14, fontweight='bold', rotation=0)

# Create MAE axis
ax2 = ax1.twinx()
bar2 = ax2.bar(index, mae, bar_width,
               color=colors['MAE'], edgecolor='black', linewidth=1.5, label='MAE')
ax2.set_ylabel('MAE Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax2.set_ylim(0, max(mae)*1.5)

# Create RMSE axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
bar3 = ax3.bar(index + bar_width, rmse, bar_width,
               color=colors['RMSE'], edgecolor='black', linewidth=1.5, label='RMSE')
ax3.set_ylabel('RMSE Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax3.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax3.set_ylim(0, max(rmse)*1.5)

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

plt.tight_layout(pad=3)
plt.savefig('D:/hsx/desktop/machine learning-literature/高清图/R2_MAE_RMSE.pdf', bbox_inches='tight')
plt.show()