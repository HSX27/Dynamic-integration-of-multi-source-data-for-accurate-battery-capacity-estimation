import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
from bayes_opt import BayesianOptimization, UtilityFunction
import time
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Set Matplotlib parameters
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False  # Handle negative sign display
})

# Load data
file_path = r"D:\hsx\desktop\machine learning-literature\工作薄(4) - 副本副本.xlsx"
df = pd.read_excel(file_path)

# Features and target
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m',
            'Current_m', 'Temp_m', 'Current_l', 'Voltage_l', 'CCCT', 'CVCT']
target = 'QDischarge'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# Check feature variance to diagnose potential issues
print("=== Feature Variance Check ===")
for feature in features:
    variance = X_train_scaled[feature].var()
    unique_count = X_train_scaled[feature].nunique()
    p05 = X_train_scaled[feature].quantile(0.05)
    p95 = X_train_scaled[feature].quantile(0.95)
    diff = p95 - p05
    print(f"{feature}: Variance = {variance:.6f}, Unique = {unique_count}, "
          f"5th = {p05:.6f}, 95th = {p95:.6f}, Diff = {diff:.6f}")
    if diff < 0.01 or unique_count <= 5:
        print(f"WARNING: {feature} may cause issues in PDP computation")

# Bayesian optimization function
def bo_params_xgb(max_depth, colsample_bytree, gamma, learning_rate, n_estimators, subsample, min_child_weight,
                  alpha, lambda_, scale_pos_weight):
    params = {
        'max_depth': int(max_depth),
        'colsample_bytree': colsample_bytree,
        'gamma': int(gamma),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'subsample': subsample,
        'min_child_weight': min_child_weight,
        'alpha': alpha,
        'reg_lambda': lambda_,  # Corrected to reg_lambda
        'scale_pos_weight': scale_pos_weight
    }
    start_time = time.time()
    reg = xgb.XGBRegressor(random_state=42, **params)
    reg.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    preds = reg.predict(X_train_scaled)
    rmse = np.sqrt(mean_squared_error(y_train, preds))
    r2 = r2_score(y_train, preds)
    print(f"Params: {params}, R2: {r2:.4f}, RMSE: {rmse:.4f}, Time: {training_time:.2f}s")
    return -rmse

# Initialize Bayesian optimization
xgb_BO = BayesianOptimization(
    f=bo_params_xgb,
    pbounds={
        'max_depth': (3, 6),
        'colsample_bytree': (0.5, 1.0),
        'gamma': (1, 5),
        'learning_rate': (0.01, 0.1),
        'n_estimators': (50, 100),
        'subsample': (0.5, 1),
        'min_child_weight': (1, 10),
        'alpha': (0.1, 1),
        'lambda_': (1, 10),
        'scale_pos_weight': (1, 10)
    },
    random_state=42
)

# Run optimization
utility_function = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
xgb_BO.maximize(init_points=5, n_iter=25, acquisition_function=utility_function)

# Train final model with best parameters
best_params = xgb_BO.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['gamma'] = int(best_params['gamma'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['reg_lambda'] = best_params.pop('lambda_')

best_xgb_model = xgb.XGBRegressor(random_state=42, **best_params)
best_xgb_model.fit(X_train_scaled, y_train)

# Test set evaluation
y_pred = best_xgb_model.predict(X_test_scaled)
print(f"Test R2 score: {r2_score(y_test, y_pred):.4f}")

# 2D Partial Dependence Plots
features_2d = [('CVCT', 'Qdlin'), ('CVCT', 'SOH'), ('CVCT', 'Temp_m')]
for feature_pair in features_2d:
    print(f'Computing 2D PDP for {feature_pair}...')
    display = PartialDependenceDisplay.from_estimator(
        best_xgb_model,
        X_train_scaled,
        features=[feature_pair],
        kind='average',
        subsample=50,
        n_jobs=1,
        grid_resolution=20,
        random_state=42,
        percentiles=(0.001, 0.999)  # Modified to a wider percentile range
    )
    display.figure_.suptitle(f'2D Partial Dependence Plot for {feature_pair}')
    plt.show()

# ICE Plots and Combined Plots are similar
# Correlation Heatmap
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

# 1D Partial Dependence Plots
selected_features = ['Temp_m','SOH','CVCT', 'Qdlin']
# Combined PDP + ICE Plots
for feature in selected_features:
    print(f'Computing PDP + ICE for {feature}...')
    display = PartialDependenceDisplay.from_estimator(
        best_xgb_model, X_train_scaled, [feature], kind="both", subsample=50,
        n_jobs=1, grid_resolution=20, random_state=42, percentiles=(0.001, 0.999)
    )
    display.figure_.suptitle(f'PDP + ICE for {feature}')
    plt.show()