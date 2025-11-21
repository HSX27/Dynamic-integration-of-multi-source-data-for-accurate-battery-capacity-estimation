import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==================== Global Font Settings (Consistent with importance value2.py) ====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Ensure negative signs display correctly

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==================== Data Preparation ====================
file_path = "/root/database.xlsx"
features = ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m', 'Current_m',
            'Temp_m', 'Current_l', 'Voltage_l', 'CCCT', 'CVCT']
target = 'QDischarge'

# Load data
data = pd.read_excel(file_path)
X = data[features].values.astype(np.float32)
y = data[target].values.astype(np.float32)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==================== Model Preparation ====================
ridge_params = {'alpha': 0.01936174112807432}
xgboost_params = {
    'n_estimators': 661,
    'max_depth': 15,
    'learning_rate': 0.018985687835643434,
    'subsample': 0.8070091419786368,
    'colsample_bytree': 0.8895164161417117,
    'gamma': 0.0007674141863846518,
    'reg_alpha': 0.4146452109077081,
    'reg_lambda': 0.8495794761874473,
    'tree_method': 'gpu_hist'
}

# Train Ridge model
ridge = Ridge(**ridge_params)
ridge.fit(X_train, y_train)

# Train XGBoost model
xgb_model = XGBRegressor(**xgboost_params)
xgb_model.fit(X_train, y_train)

# Load pre-trained LSTM model
lstm_model = tf.keras.models.load_model('/root/gpu_model_default.h5')

# Generate predictions from base models
X_train_meta = np.hstack([
    ridge.predict(X_train).reshape(-1, 1),
    xgb_model.predict(X_train).reshape(-1, 1),
    lstm_model.predict(tf.expand_dims(tf.convert_to_tensor(X_train), axis=1)).reshape(-1, 1)
])

X_test_meta = np.hstack([
    ridge.predict(X_test).reshape(-1, 1),
    xgb_model.predict(X_test).reshape(-1, 1),
    lstm_model.predict(tf.expand_dims(tf.convert_to_tensor(X_test), axis=1)).reshape(-1, 1)
])

# Train meta-model
meta_model = Ridge(alpha=0.5)
meta_model.fit(X_train_meta, y_train)

# ==================== Feature Importance Calculation ====================
# Ridge feature importance
ridge_importance = np.abs(ridge.coef_)

# XGBoost feature importance
xgb_importance = xgb_model.feature_importances_

# LSTM feature importance (permutation importance)
def permutation_importance(model, X, y, metric=mean_squared_error, n_repeats=10):
    X_3d = tf.expand_dims(tf.convert_to_tensor(X), axis=1)
    baseline = metric(y, model.predict(X_3d).flatten())
    importances = []
    for col in range(X.shape[1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, col])
            X_permuted_3d = tf.expand_dims(tf.convert_to_tensor(X_permuted), axis=1)
            score = metric(y, model.predict(X_permuted_3d).flatten())
            scores.append(score - baseline)
        importances.append(np.mean(scores))
    return np.array(importances)

lstm_importance = permutation_importance(lstm_model, X_test, y_test)

# Normalize feature importance values
ridge_importance /= ridge_importance.sum()
xgb_importance /= xgb_importance.sum()
lstm_importance /= lstm_importance.sum()

# Get meta-model coefficients and calculate stacked model feature importance
meta_coefs = meta_model.coef_
stacked_importance = (ridge_importance * meta_coefs[0] +
                      xgb_importance * meta_coefs[1] +
                      lstm_importance * meta_coefs[2])
stacked_importance /= stacked_importance.sum()

# ==================== Visualization (Key Modification Section) ====================
plt.figure(figsize=(5, 4))
sorted_idx = np.argsort(stacked_importance)
feature_colors = ['#B0B0B0', '#C6C6C6', '#DBDBDB', '#BACCC4', '#99BCAC',
                  '#95B98D', '#90B56D', '#94BF75', '#97C87D', '#A8D48B', '#B9DF99']
sorted_colors = [feature_colors[i % len(feature_colors)] for i in sorted_idx]

# Plot horizontal bar chart (Explicitly specify font)
plt.barh(range(len(sorted_idx)), stacked_importance[sorted_idx], align='center', color=sorted_colors)

# Unified font settings (Exactly the same as importance value2.py)
plt.yticks(
    range(len(sorted_idx)),
    [features[i] for i in sorted_idx],
    fontsize=8,  # Adjust tick label font size to 8
    fontfamily='Times New Roman'  # Explicitly specify font
)

plt.xlabel(
    'Normalized Importance',
    fontsize=10,  # Adjust axis label font size to 10
    fontfamily='Times New Roman'
)

plt.title(
    'Stacked Model Feature Importance',
    fontsize=12,  # Adjust title font size to 12
    pad=10,
    fontweight='bold',  # Bold title
    fontfamily='Times New Roman'
)

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()