import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from bayes_opt import BayesianOptimization
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Set Matplotlib parameters
plt.rcParams.update({'font.family': 'Times New Roman', 'axes.unicode_minus': False})

# Load data
file_path = r"D:\hsx\desktop\machine learning-literature\工作薄(4) - 副本副本.xlsx"
df = pd.read_excel(file_path)
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m', 'Current_m', 'Temp_m',
            'Current_l', 'Voltage_l', 'CCCT', 'CVCT']
target = 'QDischarge'
X = df[features]
y = df[target]

# Data splitting and standardization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)

# Bayesian optimization function
def ridge_evaluate(alpha):
    params = {'alpha': alpha}
    model = Ridge(**params)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return r2_score(y_test, y_pred)

# Parameter ranges
pbounds = {'alpha': (0.01, 10.0)}

# Perform optimization
optimizer = BayesianOptimization(f=ridge_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)
best_params = optimizer.max['params']
best_params = {'alpha': best_params['alpha']}

# Train final model
ridge_model = Ridge(**best_params)
start_time = time.time()
ridge_model.fit(X_train_scaled, y_train)
training_time = time.time() - start_time

# Evaluate model
y_pred = ridge_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Ridge - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Training time: {training_time:.2f}s")

# PDP and ICE plotting functions
def plot_pdp_ice(model, model_name, X_train_scaled):
    feature_pairs = [('Voltage_m', 'Temp_m'), ('Voltage_m', 'Voltage_l'), ('Voltage_m', 'Current_m')]
    selected_features = ['Temp_m', 'Voltage_m', 'Voltage_l', 'Current_m']
    print(f"\nGenerating plots for {model_name}...")

    for pair in feature_pairs:
        print(f"Calculating 2D PDP for {pair}...")
        display = PartialDependenceDisplay.from_estimator(
            model, X_train_scaled, [pair], kind='average', subsample=50,
            n_jobs=1, grid_resolution=20, random_state=42, percentiles=(0.001, 0.999)
        )
        display.figure_.suptitle(f'2D PDP for {pair} - {model_name}', fontsize=12)
        plt.tight_layout()
        plt.show()

    # ICE + PDP
    for feature in selected_features:
        print(f"Calculating PDP + ICE for {feature}...")
        display = PartialDependenceDisplay.from_estimator(
            model, X_train_scaled, [feature], kind='both', subsample=50,
            n_jobs=1, grid_resolution=20, random_state=42, percentiles=(0.001, 0.999)
        )
        display.figure_.suptitle(f'PDP + ICE for {feature} - {model_name}', fontsize=12)
        plt.tight_layout()
        plt.show()

# Generate PDP and ICE plots
plot_pdp_ice(ridge_model, "Ridge", X_train_scaled)

# Correlation heatmap
corr = X.corr()
pval = pd.DataFrame(index=corr.index, columns=corr.columns)
for col in corr.columns:
    for row in corr.index:
        corr_value, p_value = pearsonr(X[row], X[col])
        pval.loc[row, col] = p_value

mask = np.triu(np.ones_like(corr, dtype=bool))
with plt.rc_context({'font.family': 'Times New Roman'}):
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', linewidths=0.3, cmap='GnBu',
                     annot_kws={"size": 13, "family": "Times New Roman"})
    for i in range(len(corr.columns)):
        for j in range(i):
            if pval.iloc[i, j] < 0.05:
                plt.text(j + 0.5, i + 0.2, '*', ha='center', va='center', color='black',
                         fontsize=14, fontfamily='Times New Roman')
    plt.title("Correlation Heatmap with Significance", fontdict={'family': 'Times New Roman'})
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='Times New Roman', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='Times New Roman', fontsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_family('Times New Roman')
    plt.tight_layout()
    plt.show()