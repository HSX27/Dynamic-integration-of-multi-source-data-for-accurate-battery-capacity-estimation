#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras import mixed_precision
from tensorflow.keras.mixed_precision import LossScaleOptimizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from cuml.linear_model import Ridge  # For meta-model
from xgboost import XGBRegressor, set_config  # GPU-accelerated XGBoost
from cuml.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import optuna
import cupy as cp
import gc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pdpbox import pdp  # Updated import for pdpbox==0.2.1
# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Handle negative sign display
# ==================== Global Font Configuration ====================
plt.rcParams.update({
    'font.family': 'Times New Roman',  # Main font
    'axes.titlesize': 14,              # Title font size
    'axes.labelsize': 12,              # Axis label font size
    'xtick.labelsize': 10,             # X-axis tick font size
    'ytick.labelsize': 10,             # Y-axis tick font size
    'legend.fontsize': 10,             # Legend font size
    'figure.titlesize': 14,            # Figure title font size
    'axes.unicode_minus': False        # Resolve negative sign display issue
})

# Force override colorbar font (needs separate setting)
plt.rcParams['axes.formatter.use_mathtext'] = False  # Disable math symbol font
# Set global random seed for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# Set global XGBoost configuration
set_config(verbosity=0)

# ==================== GPU Configuration ====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[:7], 'GPU')
        for gpu in gpus[:7]:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# ==================== Training Progress Monitoring ====================
class TrainingProgress(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.pbar = None
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(total=self.total_epochs, desc='Training', unit='epoch')

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.update(1)
        self.pbar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'val_loss': f"{logs.get('val_loss', 'NA')}"
        })

    def on_train_end(self, logs=None):
        self.pbar.close()

def log_step(message):
    print(f"\n{'=' * 50}")
    print(message)
    print(f"{'=' * 50}")

# ==================== Data Preprocessing ====================
class DataProcessor:
    def __init__(self, features, target):
        self.scaler = StandardScaler()
        self.features = features
        self.target = target

    def load_and_preprocess(self, file_path):
        data = pd.read_excel(file_path)
        X = data[self.features].values.astype(np.float32)
        y = data[self.target].values.astype(np.float32)
        X = self.scaler.fit_transform(X)
        return tf.convert_to_tensor(X), tf.convert_to_tensor(y)

# ==================== Model Building ====================
class ModelBuilder:
    @staticmethod
    def build_lstm(input_shape, units=320, lr=0.0005):
        model = Sequential([
            Input(shape=(1, input_shape[0])),
            LSTM(
                units,
                activation='tanh',
                recurrent_activation='sigmoid',
                kernel_initializer='he_normal',
                unroll=False,
                dtype='mixed_float16'
            ),
            Dense(1, kernel_initializer='he_normal', dtype='float32')
        ])
        optimizer = Adam(learning_rate=lr)
        optimizer = LossScaleOptimizer(optimizer)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        return model

    @staticmethod
    def get_base_models(ridge_params, xgboost_params, lstm_params):
        return [
            ('Ridge', {'model': Ridge, 'params': ridge_params}),
            ('XGBoost', {'model': XGBRegressor, 'params': xgboost_params}),
            ('LSTM', lstm_params)
        ]

# ==================== Training Process ====================
class GPUTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_fold(self, train_idx, valid_idx, model_params):
        X_tr = tf.gather(self.X, train_idx)
        y_tr = tf.gather(self.y, train_idx)
        X_val = tf.gather(self.X, valid_idx)
        y_val = tf.gather(self.y, valid_idx)

        X_tr = tf.expand_dims(X_tr, axis=1)
        X_val = tf.expand_dims(X_val, axis=1)

        batch_size = self._calculate_batch_size(X_tr.shape, model_params)
        train_data = self._create_pipeline(X_tr, y_tr, batch_size, shuffle=True)
        val_data = self._create_pipeline(X_val, y_val, batch_size, shuffle=False)

        model = self._build_and_train_model(train_data, val_data, X_tr.shape[2], model_params)
        return model.predict(val_data).flatten()

    def _calculate_batch_size(self, shape, params):
        return min(params.get('bs', 512), 2 ** 14 // shape[1])

    def _create_pipeline(self, X, y, batch_size, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X) * 2, seed=42)  # Fixed seed
        dataset = dataset.cache()
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def _build_and_train_model(self, train_data, val_data, input_dim, params):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = ModelBuilder.build_lstm(
                input_shape=(input_dim,),
                units=params['units'],
                lr=params['lr']
            )
        checkpoint_name = params.get("name", "default")
        model.fit(
            train_data,
            validation_data=val_data,
            epochs=200,
            verbose=0,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ModelCheckpoint(f'gpu_model_{checkpoint_name}.h5', save_best_only=True),
                TrainingProgress(200)
            ]
        )
        return model

# ==================== Main Process ====================
def stacking_workflow(file_path, features, target):
    # Log initialization
    log_step("Initializing Data Processor")
    processor = DataProcessor(features, target)
    X, y = processor.load_and_preprocess(file_path)
    X = X.numpy()
    y = y.numpy()

    # Predefined optimal hyperparameters
    ridge_params = {'alpha': 0.01936174112807432}
    xgboost_params = {
        'n_estimators': 661, 'max_depth': 15, 'learning_rate': 0.018985687835643434,
        'subsample': 0.8070091419786368, 'colsample_bytree': 0.8895164161417117,
        'gamma': 0.0007674141863846518, 'reg_alpha': 0.4146452109077081,
        'reg_lambda': 0.8495794761874473, 'tree_method': 'gpu_hist'
    }
    lstm_params = {'units': 320, 'lr': 0.00033573557244561997, 'bs': 128}

    log_step("Building Ensemble Models")
    global base_models, meta_model, X_train, y_train, X_test, base_models_trained  # Declare as global variables for prediction function
    base_models = ModelBuilder.get_base_models(ridge_params, xgboost_params, lstm_params)

    # Split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train base models and save trained models
    base_models_trained = []
    for name, model_config in base_models:
        if name == 'LSTM':
            model = tf.keras.models.load_model('/root/gpu_model_default.h5')  # Load pretrained LSTM model
        else:
            model = model_config['model'](**model_config['params'])
            if name == 'Ridge':
                model.fit(cp.asarray(X_train), cp.asarray(y_train))
            else:  # XGBoost
                model.fit(X_train, y_train)
        base_models_trained.append((name, model))

    # Train meta-model
    base_preds_train = []
    for name, model in base_models_trained:
        if name == 'LSTM':
            X_train_3d = tf.expand_dims(tf.convert_to_tensor(X_train.astype(np.float32)), axis=1)
            train_pred = model.predict(X_train_3d, verbose=0).flatten()
        elif name == 'Ridge':
            train_pred = model.predict(cp.asarray(X_train)).get()
        else:  # XGBoost
            train_pred = model.predict(X_train)
        base_preds_train.append(train_pred.reshape(-1, 1))

    meta_X_train = np.hstack(base_preds_train)
    meta_model = Ridge(alpha=0.5)
    meta_model.fit(meta_X_train.astype(np.float32), y_train.astype(np.float32))

    # Generate final predictions on test set
    base_preds_test = []
    for name, model in base_models_trained:
        if name == 'LSTM':
            X_test_3d = tf.expand_dims(tf.convert_to_tensor(X_test.astype(np.float32)), axis=1)
            test_pred = model.predict(X_test_3d, verbose=0).flatten()
        elif name == 'Ridge':
            test_pred = model.predict(cp.asarray(X_test)).get()
        else:  # XGBoost
            test_pred = model.predict(X_test)
        base_preds_test.append(test_pred.reshape(-1, 1))

    meta_X_test = np.hstack(base_preds_test)
    final_pred = meta_model.predict(meta_X_test.astype(np.float32))

    # Evaluate model
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, final_pred)),
        'MAE': mean_absolute_error(y_test, final_pred),
        'R2': r2_score(y_test, final_pred)
    }
    print(f"Evaluation Metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")

    # Save predictions and true values
    pd.DataFrame({'True': y_test, 'Pred': final_pred}).to_csv('predictions.csv', index=False)
    log_step("Predictions and true values saved to 'predictions.csv'")

    # Define stacking model prediction function (batch prediction)
    def stacking_predict(X):
        base_preds = []
        for name, model in base_models_trained:
            if name == 'LSTM':
                X_3d = tf.expand_dims(tf.convert_to_tensor(X.astype(np.float32)), axis=1)
                pred = model.predict(X_3d, verbose=0).flatten()
            elif name == 'Ridge':
                pred = model.predict(cp.asarray(X)).get()
            else:  # XGBoost
                pred = model.predict(X)
            base_preds.append(pred.reshape(-1, 1))
        meta_X = np.hstack(base_preds)
        final_pred = meta_model.predict(meta_X.astype(np.float32))
        return final_pred

    # Generate PDP and 2D PDP plots
    class StackingModelWrapper:
        def __init__(self, predict_func):
            self.predict = predict_func

    stacking_model = StackingModelWrapper(stacking_predict)
    X_test_df = pd.DataFrame(X_test, columns=features)
    selected_features = ['CVCT', 'Temp_m', 'Current_m', 'SOH']

    # ==================== Generate 2D Interaction Partial Dependence Plots ====================
    log_step("Generating 2D Interaction Partial Dependence Plots")
    feature_pairs = [('CVCT', 'Current_m'), ('CVCT', 'Temp_m'), ('CVCT', 'SOH')]

    for feature1, feature2 in feature_pairs:
        # Generate interaction data
        pdp_interact_data = pdp.pdp_interact(
            model=stacking_model,
            dataset=X_test_df,
            model_features=features,
            features=[feature1, feature2],
            num_grid_points=[20, 20]
        )

        # Create figure object
        plt.figure(figsize=(10, 8))

        # Step 1: Get the plot objects
        fig, ax_dict = pdp.pdp_interact_plot(
            pdp_interact_data,
            feature_names=[feature1, feature2],
            plot_type='contour',
            plot_params={
                'contour_color': 'black',
                'contour_linewidths': 0.5,
                'fill_alpha': 0.7,
                'cmap': 'viridis',
                'colorbar': False
            }
        )

        # Step 2: Force rendering
        fig.canvas.draw()

        # Step 3: Get the axes object dynamically
        ax = next(iter(ax_dict.values()))  # Access the first axes object

        # Step 4: Add custom colorbar if collections exist
        if ax.collections:
            contour = ax.collections[0]
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label("Predicted QDischarge", fontname='Times New Roman', fontsize=12)
            cbar.ax.tick_params(labelsize=10, labelrotation=45)
        else:
            print(f"No collections found for {feature1} and {feature2}")

        # Step 5: Set fonts and labels
        ax.set_title(
            f"2D PDP: Interaction between {feature1} and {feature2}",
            fontdict={'fontname': 'Times New Roman', 'fontsize': 14},
            pad=10
        )
        ax.set_xlabel(feature1, fontname='Times New Roman', fontsize=12)
        ax.set_ylabel(feature2, fontname='Times New Roman', fontsize=12)

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontname('Times New Roman')

        if 'cbar' in locals():
            for text in cbar.ax.get_yticklabels():
                text.set_fontname('Times New Roman')

        plt.tight_layout()
        plt.show()
    # ==================== Generate PDP and ICE Plots ====================
    log_step("Generating Partial Dependence and ICE Plots")
    selected_features = ['CVCT', 'Temp_m', 'Current_m', 'SOH']

    for feature in selected_features:
        pdp_data = pdp.pdp_isolate(
            model=stacking_model,
            dataset=X_test_df,
            model_features=features,
            feature=feature,
            num_grid_points=20
        )
        plt.figure(figsize=(10, 6))
        pdp.pdp_plot(
            pdp_data,
            feature,
            plot_lines=True,
            center=False,
            plot_params={
                'pdp_color': '#1A4E66',
                'pdp_linewidth': 2.5,
                'ice_color': '#A9CCE3',
                'ice_alpha': 0.3,
                'font_family': 'Times New Roman'  # Explicitly specify font
            }
        )
        ax = plt.gca()
        ax.set_title(f"PDP and ICE Plot for {feature}", fontname='Times New Roman', fontsize=14)
        ax.set_xlabel(feature, fontname='Times New Roman', fontsize=12)
        ax.set_ylabel("Predicted QDischarge", fontname='Times New Roman', fontsize=12)
        plt.xticks(fontname='Times New Roman', fontsize=10)
        plt.yticks(fontname='Times New Roman', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

# Example execution
if __name__ == "__main__":
    config = {
        "file_path": "/root/database.xlsx",
        "features": ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m', 'Current_m', 'Temp_m', 'Current_l', 'Voltage_l',
                     'CCCT', 'CVCT'],
        "target": 'QDischarge'
    }
    results = stacking_workflow(**config)
    print("\nFinal Evaluation Metrics:")
    print(f"- RMSE: {results['RMSE']:.4f}\n- MAE: {results['MAE']:.4f}\n- R²: {results['R2']:.4f}")