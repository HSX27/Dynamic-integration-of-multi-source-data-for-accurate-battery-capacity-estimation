import matplotlib.pyplot as plt
import numpy as np

# Data preparation:
conditions = ['SE', 'KNN', 'RF', 'Ridge', 'NN', 'XGB', 'LGBM', 'MLP', 'SVM']
f1 = [0.4363, 0.3874, 0.2832, 0.5043, 0.9269, 0.2792, 0.3438, 0.5827, 0.3450]
f2 = [0.2881, 0.2863, 0.2242, 0.4535, 0.0408, 0.2107, 0.1802, 0.1168, 0.0890]
f3 = [0.1358, 0.1724, 0.1654, 0.0168, 0.0164, 0.1153, 0.1306, 0.0589, 0.0890]

# Plot parameter settings
bar_width = 0.25
index = np.arange(len(conditions))
colors = {
    'F1': '#0881a3',
    'F2': '#ffd6a4',
    'F3': '#fde9df'
}

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot F1 bar chart
bar1 = ax1.bar(index - bar_width, f1, bar_width,
               color=colors['F1'], edgecolor='black', linewidth=1.5, label='F1')
ax1.set_xlabel('Model', fontsize=16, fontweight='bold')
ax1.set_ylabel('F1 Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax1.set_ylim(0.2, 1.0)
ax1.set_xticks(index)
ax1.set_xticklabels(conditions, fontsize=14, fontweight='bold', rotation=0)

# Create F2 axis
ax2 = ax1.twinx()
bar2 = ax2.bar(index, f2, bar_width,
               color=colors['F2'], edgecolor='black', linewidth=1.5, label='F2')
ax2.set_ylabel('F2 Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax2.set_ylim(0, max(f2)*1.8)

# Create F3 axis
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
bar3 = ax3.bar(index + bar_width, f3, bar_width,
               color=colors['F3'], edgecolor='black', linewidth=1.5, label='F3')
ax3.set_ylabel('F3 Value', fontsize=16, fontweight='bold', color='black')  # 修改为黑色
ax3.tick_params(axis='y', labelcolor='black', labelsize=14)  # 修改为黑色
ax3.set_ylim(0, max(f3)*1.8)

# Unify axis styles
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(True)
    for side in ['left', 'right', 'bottom', 'top']:
        ax.spines[side].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=14)

# Adjust legend position and style
ax1.legend([bar1[0], bar2[0], bar3[0]], ['F1', 'F2', 'F3'],
           loc='upper left', fontsize=12, frameon=False,
           bbox_to_anchor=(0.83, 0.98),
           handletextpad=0.5, columnspacing=1.2)

plt.tight_layout(pad=3)
plt.savefig('D:/hsx/desktop/machine learning-literature/高清图/F.pdf', bbox_inches='tight')
plt.show()