import numpy as np
import pandas as pd
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

# Set Matplotlib parameters
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False  # Handle negative sign display
})

# Set random seed for reproducibility
np.random.seed(42)

# Load data
file_path = r"D:\hsx\desktop\machine learning-literature\工作薄(4) - 副本副本.xlsx"
df = pd.read_excel(file_path)

# Data inspection
print(f"Dataset shape: {df.shape}")
print(f"Missing values check:\n{df.isnull().sum()}")

# Define features and target variable
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m',
            'Current_m', 'Temp_m', 'Current_l', 'Voltage_l', 'CCCT', 'CVCT']
target = 'QDischarge'  # Assume the target is QDischarge

X = df[features]
y = df[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter ranges and transformation functions for Bayesian optimization
bo_params_funcs = {
    'XGBoost': {
        'function': lambda max_depth, colsample_bytree, gamma, learning_rate, n_estimators, subsample, min_child_weight,
                           alpha, lambda_, scale_pos_weight: {
            'max_depth': int(max_depth),
            'colsample_bytree': colsample_bytree,
            'gamma': int(gamma),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'subsample': subsample,
            'min_child_weight': min_child_weight,
            'alpha': alpha,
            'lambda': lambda_,
            'scale_pos_weight': scale_pos_weight
        },
        'bounds': {
            'max_depth': (3, 6),  # Adjust to smaller values
            'colsample_bytree': (0.5, 1),  # Reduce sampling proportion
            'gamma': (1, 5),  # Increase splitting threshold
            'learning_rate': (0.01, 0.1),  # Reduce learning rate
            'n_estimators': (50, 100),  # Reduce number of trees
            'subsample': (0.5, 1),  # Reduce sampling proportion
            'min_child_weight': (1, 10),
            'alpha': (0.1, 1),  # Increase regularization
            'lambda_': (1, 10),  # Increase regularization
            'scale_pos_weight': (1, 10)
        }
    },
    'LightGBM': {
        'function': lambda max_depth, learning_rate, n_estimators, subsample, min_child_samples, num_leaves: {
            'max_depth': int(max_depth),
            'learning_rate': learning_rate,
            'n_estimators': int(n_estimators),
            'subsample': subsample,
            'min_child_samples': int(min_child_samples),
            'num_leaves': int(num_leaves)
        },
        'bounds': {
            'max_depth': (3, 5),  # Adjust to smaller values
            'learning_rate': (0.01, 0.05),  # Reduce learning rate
            'n_estimators': (30, 50),  # Reduce number of trees
            'subsample': (0.5, 0.8),  # Reduce sampling proportion
            'min_child_samples': (5, 10),
            'num_leaves': (20, 30)  # Reduce number of leaf nodes
        }
    },
    'MLP': {
        'function': lambda hidden_layer_sizes, learning_rate_init, alpha, max_iter: {
            'hidden_layer_sizes': (int(hidden_layer_sizes),),
            'learning_rate_init': learning_rate_init,
            'alpha': alpha,
            'max_iter': int(max_iter)
        },
        'bounds': {
            'hidden_layer_sizes': (10, 50),  # Reduce hidden layer size
            'learning_rate_init': (0.001, 0.01),  # Adjust learning rate
            'alpha': (0.1, 1),  # Increase regularization
            'max_iter': (100, 200)  # Reduce number of iterations
        }
    }
}

# Initialize models
models = {
    'XGBoost': xgb.XGBRegressor(random_state=42),
    'LightGBM': lgb.LGBMRegressor(random_state=42),
    'SVM': SVR(),
    'MLP': MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
}

# Store model results
model_results = {}

# Define Bayesian optimization function
def optimize_model(model_name, model, params_func, bounds, init_points=5, n_iter=25):
    def target(**kwargs):
        params = params_func(**kwargs)
        model.set_params(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        return -rmse  # Minimize RMSE

    optimizer = BayesianOptimization(f=target, pbounds=bounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max['params']

# Early stopping parameters
early_stopping_params = {
    'XGBoost': {'early_stopping_rounds': 10},
    'LightGBM': {'early_stopping_rounds': 10},
}

# Train and tune models
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    if model_name in bo_params_funcs:
        param_func = bo_params_funcs[model_name]['function']
        bounds = bo_params_funcs[model_name]['bounds']
        best_params = optimize_model(model_name, model, param_func, bounds)
        print(f"{model_name} best parameters: {best_params}")
        best_params = param_func(**best_params)
        model.set_params(**best_params)

    if model_name in early_stopping_params:
        model.set_params(**early_stopping_params[model_name])

    start_time = time.time()
    if model_name in ['XGBoost', 'LightGBM']:
        model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)])
    else:
        model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Prediction and evaluation
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_results[model_name] = {'Model': model, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Training Time': training_time}
    print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Training Time: {training_time:.2f}s")

# --------------------------- Feature Importance Visualization ---------------------------
feature_colors = ['#B0B0B0', '#C6C6C6', '#DBDBDB', '#BACCC4', '#99BCAC',
                  '#95B98D', '#90B56D', '#94BF75', '#97C87D', '#A8D48B', '#B9DF99']

plt.figure(figsize=(12, 10))
for i, (model_name, result) in enumerate(model_results.items()):
    model = result['Model']
    ax = plt.subplot(2, 2, i + 1)

    # Calculate feature importance
    if model_name == 'XGBoost':
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type='total_gain')
        importances = np.array([importance_dict.get(f'f{i}', 0) for i in range(len(features))])
    elif model_name == 'LightGBM':
        importances = model.booster_.feature_importance(importance_type='split')
    else:
        result_perm = permutation_importance(model, X_test_scaled, y_test, n_repeats=10,
                                             scoring='neg_mean_squared_error', random_state=42)
        importances = result_perm.importances_mean

    # Normalize importance values
    total = np.sum(importances)
    if total != 0:
        importances = importances / total
    else:
        importances = np.zeros_like(importances)

    # Get top three important features and their importance values
    sorted_idx_desc = np.argsort(importances)[::-1][:3]  # Sort in descending order and take top three
    top_features = [features[idx] for idx in sorted_idx_desc]  # Names of top three features
    top_importances = importances[sorted_idx_desc]  # Importance values of top three features

    # Print top three features and their importance values
    print(f"\nTop three important features for {model_name}:")
    for feature, importance in zip(top_features, top_importances):
        print(f"{feature}: {importance:.4f}")

    # Sort for plotting
    sorted_idx = np.argsort(importances)  # Sort in ascending order for plotting
    sorted_colors = [feature_colors[idx] for idx in sorted_idx]

    # Plot horizontal bar chart
    ax.barh(range(len(features)), importances[sorted_idx], align='center', color=sorted_colors)

    # Set plot title and labels
    ax.set_title(f'{model_name} Feature Importance',
                 fontsize=12, pad=10,
                 fontfamily='Times New Roman',
                 fontweight='bold')
    ax.set_xlabel('Normalized Importance',
                  fontsize=10,
                  fontfamily='Times New Roman')
    ax.set_ylabel('Features',
                  fontsize=10,
                  fontfamily='Times New Roman')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[j] for j in sorted_idx],
                       fontsize=8,
                       fontfamily='Times New Roman')

    # Other settings
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
plt.show()