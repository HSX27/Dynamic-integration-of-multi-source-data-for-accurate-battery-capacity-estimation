import os
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import matplotlib
import tensorflow as tf

# Disable GPU and enforce CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Check available GPUs (optional)
print("Number of GPUs available: ", len(tf.config.list_physical_devices('GPU')))

# Set global font for plotting
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['axes.unicode_minus'] = False  # Ensure negative signs display correctly

# Set random seed for reproducibility
np.random.seed(42)

# Load dataset (update file path as needed)
file_path = r"D:\hsx\desktop\machine learning-literature\工作薄(4) - 副本副本.xlsx"
df = pd.read_excel(file_path)

# Data inspection
print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Define features and target variable
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m',
            'Current_m', 'Temp_m', 'Current_l', 'Voltage_l', 'CCCT', 'CVCT']
target = 'QDischarge'

# Prepare data
X = df[features]
y = df[target]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bayesian optimization parameter settings
bo_params_funcs = {
    'RandomForest': {
        'function': lambda max_depth, n_estimators, min_samples_split, max_features: {
            'max_depth': int(max_depth),
            'n_estimators': int(n_estimators),
            'min_samples_split': int(min_samples_split),
            'max_features': max_features
        },
        'bounds': {
            'max_depth': (2, 5),
            'n_estimators': (10, 50),
            'min_samples_split': (5, 20),
            'max_features': (0.1, 0.5)
        }
    },
    'KNN': {
        'function': lambda n_neighbors: {'n_neighbors': int(n_neighbors)},
        'bounds': {'n_neighbors': (150, 200)}
    },
    'Ridge': {
        'function': lambda alpha: {'alpha': alpha},
        'bounds': {'alpha': (0.01, 10.0)}
    }
}

# Initialize models
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'KNN': KNeighborsRegressor(),
    'Ridge': Ridge()
}

# Store model results
model_results = {}

# Bayesian optimization function
def optimize_model(model_name, model, params_func, bounds, init_points=5, n_iter=25):
    def target(**kwargs):
        params = params_func(**kwargs)
        model.set_params(**params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        return scores.mean()

    optimizer = BayesianOptimization(f=target, pbounds=bounds, random_state=42)
    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    return optimizer.max['params']

# First train Keras neural network (enforce CPU usage)
print("\nTraining Keras neural network...")
with tf.device('/cpu:0'):
    nn_model = Sequential([
        Input(shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 5-fold cross-validation for Keras neural network
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        nn_model.fit(X_tr, y_tr, epochs=100, batch_size=10,
                     validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
        y_pred_val = nn_model.predict(X_val, verbose=0).flatten()
        r2_scores.append(r2_score(y_val, y_pred_val))
    print(f"Keras NN - 5-fold cross-validation R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    # Final training
    start_time = time.time()
    nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=10,
                 validation_split=0.2, callbacks=[early_stopping], verbose=0)
    nn_training_time = time.time() - start_time

    # Evaluate neural network
    y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)
    model_results['Keras Neural Network'] = {
        'Model': nn_model,
        'RMSE': rmse_nn,
        'MAE': mae_nn,
        'R²': r2_nn,
        'Training Time': nn_training_time
    }

    # Training set R² for neural network
    train_pred_nn = nn_model.predict(X_train_scaled, verbose=0).flatten()
    train_r2_nn = r2_score(y_train, train_pred_nn)
    print(f"Keras NN - Training set R²: {train_r2_nn:.4f}, Test set R²: {r2_nn:.4f}")

# Train and evaluate scikit-learn models (order: RandomForest -> KNN -> Ridge)
for model_name in ['RandomForest', 'KNN', 'Ridge']:
    print(f"\nTraining {model_name}...")
    if model_name in bo_params_funcs:
        param_func = bo_params_funcs[model_name]['function']
        bounds = bo_params_funcs[model_name]['bounds']
        best_params = optimize_model(model_name, models[model_name], param_func, bounds)
        print(f"{model_name} best parameters: {best_params}")
        best_params = param_func(**best_params)
        models[model_name].set_params(**best_params)

    start_time = time.time()
    models[model_name].fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    # Evaluate on test set
    y_pred = models[model_name].predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 5-fold cross-validation
    cv_scores = cross_val_score(models[model_name], X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"{model_name} - 5-fold cross-validation R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    model_results[model_name] = {
        'Model': models[model_name],
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Training Time': training_time
    }

    # Training set R²
    train_pred = models[model_name].predict(X_train_scaled)
    train_r2 = r2_score(y_train, train_pred)
    print(f"{model_name} - Training set R²: {train_r2:.4f}, Test set R²: {r2:.4f}")

# Print all model results
print("\nAll model results:")
for name, result in model_results.items():
    print(
        f"{name}: RMSE={result['RMSE']:.4f}, MAE={result['MAE']:.4f}, R²={result['R²']:.4f}, Training Time={result['Training Time']:.2f}s")

# --------------------------- Feature Importance Visualization ---------------------------
feature_colors = ['#B0B0B0', '#C6C6C6', '#DBDBDB', '#BACCC4', '#99BCAC',
                  '#95B98D', '#90B56D', '#94BF75', '#97C87D', '#A8D48B', '#B9DF99']

plt.figure(figsize=(12, 10))
for i, (model_name, result) in enumerate(model_results.items()):
    model = result['Model']
    ax = plt.subplot(2, 2, i + 1)

    # Calculate feature importance
    if hasattr(model, 'feature_importances_'):  # RandomForest
        importances = model.feature_importances_
    else:  # KNN, Ridge, and Keras NN use permutation importance
        result_perm = permutation_importance(
            model, X_test_scaled, y_test, n_repeats=10,
            scoring='neg_mean_squared_error', random_state=42
        )
        importances = result_perm.importances_mean

    # Normalize importance values
    total = np.sum(importances)
    if total != 0:
        importances = importances / total
    else:
        importances = np.zeros_like(importances)

    # Extract top three important features
    sorted_idx_desc = np.argsort(importances)[::-1][:3]  # Sort in descending order and take top three
    top_features = [features[idx] for idx in sorted_idx_desc]  # Names of top three features
    top_importances = importances[sorted_idx_desc]  # Importance values of top three features

    # Print top three features and their importance values
    print(f"\nTop three important features for {model_name}:")
    for feature, importance in zip(top_features, top_importances):
        print(f"{feature}: {importance:.4f}")

    # Plotting (maintain existing logic)
    sorted_idx = np.argsort(importances)  # Sort in ascending order for plotting
    sorted_colors = [feature_colors[idx] for idx in sorted_idx]
    ax.barh(range(len(features)), importances[sorted_idx], align='center', color=sorted_colors)
    ax.set_title(f'{model_name} Feature Importance', fontsize=12, pad=10, fontweight='bold')
    ax.set_xlabel('Normalized Importance', fontsize=10)
    ax.set_ylabel('Features', fontsize=10)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[j] for j in sorted_idx], fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()