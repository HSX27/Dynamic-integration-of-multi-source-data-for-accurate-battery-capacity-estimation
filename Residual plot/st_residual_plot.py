import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==================== Global Font Settings ====================
plt.rcParams['font.family'] = 'Times New Roman'  # Set global font
plt.rcParams['axes.unicode_minus'] = False       # Fix negative sign display
sns.set_theme(style='whitegrid', font='Times New Roman')  # Force Seaborn to use specified font

# ==================== Data Loading and Processing ====================
file_path = r"D:\hsx\desktop\machine learning-literature\predictions.csv"
df = pd.read_csv(file_path)
true_values = df['True']
pred_values = df['Pred']

# ==================== Calculate Evaluation Metrics ====================
mae = mean_absolute_error(true_values, pred_values)
mse = mean_squared_error(true_values, pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(true_values, pred_values)

# ==================== Full Range Scatter Plot ====================
plt.figure(figsize=(8, 6))
kde = sns.kdeplot(
    x=pred_values, y=true_values,
    cmap="rainbow", fill=True,
    alpha=0.7
)
min_val = min(min(pred_values), min(true_values))
max_val = max(max(pred_values), max(true_values))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="1:1 line")

# Add metric text
plt.text(
    min(pred_values), max(true_values) * 0.95,
    f'MAE = {mae:.5f}',
    fontsize=12,
    fontname='Times New Roman'  # Force specify
)
plt.text(
    min(pred_values), max(true_values) * 0.90,
    f'MSE = {mse:.5f}',
    fontsize=12,
    fontname='Times New Roman'
)
plt.text(
    min(pred_values), max(true_values) * 0.85,
    f'RMSE = {rmse:.5f}',
    fontsize=12,
    fontname='Times New Roman'
)
plt.text(
    min(pred_values), max(true_values) * 0.80,
    f'R² = {r2:.5f}',
    fontsize=12,
    fontname='Times New Roman'
)

# Set title and labels
plt.title('Actual vs Predicted Values', fontname='Times New Roman', fontsize=14)
plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=12)
plt.ylabel('Actual Values', fontname='Times New Roman', fontsize=12)
plt.legend(prop={'family': 'Times New Roman'})  # Legend font
plt.colorbar(kde.collections[0], label='Probability Density')
plt.show()

# ==================== Zoomed-In Scatter Plot ====================
plt.figure(figsize=(8, 6))
kde_zoomed = sns.kdeplot(
    x=pred_values, y=true_values,
    cmap="rainbow", fill=True,
    alpha=0.7
)
plt.plot([1.45, 1.75], [1.45, 1.75], 'k--', lw=2, label="1:1 line")
plt.xlim(1.45, 1.75)
plt.ylim(1.45, 1.75)

# Set title and labels
plt.title('Actual vs Predicted Values (Zoomed In)', fontname='Times New Roman', fontsize=14)
plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=12)
plt.ylabel('Actual Values', fontname='Times New Roman', fontsize=12)
plt.legend(prop={'family': 'Times New Roman'})  # Legend font
plt.colorbar(kde_zoomed.collections[0], label='Probability Density')
plt.show()