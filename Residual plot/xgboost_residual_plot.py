import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import time

# ==================== Global Font Settings ====================
plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体
plt.rcParams['axes.unicode_minus'] = False       # 修复负号显示问题
sns.set_theme(style='whitegrid', font='Times New Roman')  # 设置Seaborn主题和字体

# Load data
file_path = r"D:\hsx\desktop\machine learning-literature\工作薄(4) - 副本副本.xlsx"
df = pd.read_excel(file_path)

# Features and target
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m',
            'Current_m', 'Temp_m', 'Current_l', 'Voltage_l',
            'CCCT', 'CVCT']
target = 'QDischarge'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert standardized data back to DataFrame and retain feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

# Define Bayesian optimization function
def bo_params_xgb(max_depth, colsample_bytree, gamma, learning_rate, n_estimators, subsample, min_child_weight,
                  alpha, reg_lambda, scale_pos_weight):
    params = {
        'max_depth': int(max_depth),
        'colsample_bytree': colsample_bytree,
        'gamma': int(gamma),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'subsample': subsample,
        'min_child_weight': min_child_weight,
        'alpha': alpha,
        'reg_lambda': reg_lambda,
        'scale_pos_weight': scale_pos_weight
    }

    # Start timing
    start_time = time.time()

    reg = xgb.XGBRegressor(random_state=42, **params)
    reg.fit(X_train_scaled, y_train)

    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    preds = reg.predict(X_train_scaled)
    mse = mean_squared_error(y_train, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, preds)

    print(f"Params: {params}, R2: {r2}, RMSE: {rmse}, Training time: {training_time:.2f} seconds")
    return -rmse

# Initialize Bayesian optimizer
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
        'reg_lambda': (1, 10),
        'scale_pos_weight': (1, 10)
    },
    random_state=42
)

# Define utility function
utility_function = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

# Perform Bayesian optimization using utility function
xgb_BO.maximize(init_points=5, n_iter=25, acquisition_function=utility_function)

# Get optimal parameters and train final model
best_params = xgb_BO.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['gamma'] = int(best_params['gamma'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# Train optimal XGBoost model
best_xgb_model = xgb.XGBRegressor(random_state=42, **best_params)
best_xgb_model.fit(X_train_scaled, y_train)

# Predict results
y_pred_train = best_xgb_model.predict(X_train_scaled)
y_pred_test = best_xgb_model.predict(X_test_scaled)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r_value = r2_score(y_test, y_pred_test)

# Plot scatter plot and 1:1 reference line
plt.figure(figsize=(8, 6))

# Plot scatter plot with color density to indicate point concentration
kde = sns.kdeplot(x=y_pred_test, y=y_test, cmap="rainbow", fill=True)

# Plot 1:1 reference line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label="1:1 line")

# Add error metrics
plt.text(min(y_test), max(y_test) * 0.95, f'MAE = {mae:.5f}', fontsize=12, fontname='Times New Roman')
plt.text(min(y_test), max(y_test) * 0.90, f'MSE = {mse:.5f}', fontsize=12, fontname='Times New Roman')
plt.text(min(y_test), max(y_test) * 0.85, f'RMSE = {rmse:.5f}', fontsize=12, fontname='Times New Roman')
plt.text(min(y_test), max(y_test) * 0.80, f'R = {r_value:.5f}', fontsize=12, fontname='Times New Roman')

# Set title and labels
plt.title('Actual vs Predicted Values', fontname='Times New Roman', fontsize=14)
plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=12)
plt.ylabel('Actual Values', fontname='Times New Roman', fontsize=12)

# Show legend
plt.legend(prop={'family': 'Times New Roman'})

# Show colorbar linked to kde plot
plt.colorbar(kde.collections[0], label='Probability Density')

# Display plot
plt.show()

# Create a new plot with X and Y axis ranges limited to 1.45 to 1.75
plt.figure(figsize=(8, 6))

# Plot zoomed-in scatter plot with color density to indicate point concentration
kde_zoomed = sns.kdeplot(x=y_pred_test, y=y_test, cmap="rainbow", fill=True)

# Plot 1:1 reference line
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label="1:1 line")

# Set title and labels
plt.title('Actual vs Predicted Values (Zoomed In)', fontname='Times New Roman', fontsize=14)
plt.xlabel('Predicted Values', fontname='Times New Roman', fontsize=12)
plt.ylabel('Actual Values', fontname='Times New Roman', fontsize=12)

# Set X and Y axis ranges
plt.xlim(1.45, 1.75)
plt.ylim(1.45, 1.75)

# Show legend
plt.legend(prop={'family': 'Times New Roman'})

# Show colorbar linked to kde plot
plt.colorbar(kde_zoomed.collections[0], label='Probability Density')

# Display zoomed-in plot
plt.show()