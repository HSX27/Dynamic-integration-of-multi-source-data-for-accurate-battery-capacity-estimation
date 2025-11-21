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
from xgboost import XGBRegressor, set_config  # Use GPU-accelerated XGBoost
from cuml.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import optuna
import cupy as cp
import gc
from sklearn.model_selection import train_test_split

# Set global random seed for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seeds(42)

# Set global XGBoost configuration
set_config(verbosity=0)

# ==================== GPU Environment Configuration ====================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[:4], 'GPU')
        for gpu in gpus[:4]:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
tf.config.optimizer.set_jit(True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# ==================== Progress Monitoring Component ====================
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

# ==================== Model Building Component ====================
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

# ==================== Training Process Component ====================
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

# ==================== Hyperparameter Optimization Component ====================
class HyperparameterOptimizer:
    @staticmethod
    def optimize_ridge(X, y):
        def objective(trial):
            params = {'alpha': trial.suggest_float("alpha", 1e-3, 10.0, log=True)}
            model = Ridge(**params)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            X_cp = cp.asarray(X)
            y_cp = cp.asarray(y)
            for train_idx, valid_idx in kf.split(X):
                model.fit(X_cp[train_idx], y_cp[train_idx])
                preds = model.predict(X_cp[valid_idx]).get()
                mse = mean_squared_error(y[valid_idx], preds)
                scores.append(mse)
            return np.mean(scores)

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50)
        return study.best_params

    @staticmethod
    def optimize_xgboost(X, y):
        def objective(trial):
            params = {
                'tree_method': 'gpu_hist',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'verbosity': 0,
                'random_state': 42  # Fixed random seed
            }
            model = XGBRegressor(**params)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            scores = []
            for train_idx, valid_idx in kf.split(X):
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[valid_idx])
                mse = mean_squared_error(y[valid_idx], preds)
                scores.append(mse)
            return np.mean(scores)

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50)
        return study.best_params

    @staticmethod
    def optimize_lstm(X, y):
        def objective(trial):
            params = {
                'units': trial.suggest_int('units', 128, 512, step=64),
                'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
                'bs': trial.suggest_categorical('bs', [128, 256, 512]),
                'name': f'trial_{trial.number}'
            }
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mse_scores = []
            for train_idx, valid_idx in kf.split(X):
                X_train_fold = tf.convert_to_tensor(X[train_idx].astype(np.float32))
                X_train_fold = tf.expand_dims(X_train_fold, axis=1)
                y_train_fold = tf.convert_to_tensor(y[train_idx].astype(np.float32))
                X_valid_fold = tf.convert_to_tensor(X[valid_idx].astype(np.float32))
                X_valid_fold = tf.expand_dims(X_valid_fold, axis=1)
                y_valid_fold = tf.convert_to_tensor(y[valid_idx].astype(np.float32))

                train_dataset = tf.data.Dataset.from_tensor_slices((X_train_fold, y_train_fold))
                train_dataset = train_dataset.shuffle(buffer_size=len(X_train_fold) * 2, seed=42).batch(params['bs'])
                valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid_fold, y_valid_fold)).batch(params['bs'])

                model = ModelBuilder.build_lstm(
                    input_shape=(X_train_fold.shape[2],),
                    units=params['units'],
                    lr=params['lr']
                )
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.TerminateOnNaN()
                ]
                try:
                    model.fit(
                        train_dataset,
                        validation_data=valid_dataset,
                        epochs=50,
                        verbose=0,
                        callbacks=callbacks
                    )
                    preds = model.predict(valid_dataset).flatten()
                    mse = mean_squared_error(y_valid_fold.numpy(), preds)
                    mse_scores.append(mse)
                except tf.errors.ResourceExhaustedError:
                    return float('inf')
                finally:
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
                    tf.compat.v1.reset_default_graph()
            return np.mean(mse_scores)

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50)
        return study.best_params

# ==================== Evaluation Function ====================
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

# ==================== Main Workflow ====================
def stacking_workflow(file_path, features, target):
    # Logging
    log_step("Initializing data processor")
    processor = DataProcessor(features, target)
    X, y = processor.load_and_preprocess(file_path)
    X = X.numpy()
    y = y.numpy()

    # Use predefined optimal hyperparameters
    ridge_params = {'alpha': 0.01936174112807432}
    xgboost_params = {
        'n_estimators': 661, 'max_depth': 15, 'learning_rate': 0.018985687835643434,
        'subsample': 0.8070091419786368, 'colsample_bytree': 0.8895164161417117,
        'gamma': 0.0007674141863846518, 'reg_alpha': 0.4146452109077081,
        'reg_lambda': 0.8495794761874473
    }
    lstm_params = {'units': 320, 'lr': 0.00033573557244561997, 'bs': 128}

    log_step("Building ensemble model")
    base_models = ModelBuilder.get_base_models(ridge_params, xgboost_params, lstm_params)

    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define K-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Store weights for each base model
    model_weights = {}

    # Train base models and calculate dynamic weights
    log_step("Calculating dynamic weights for base models")
    for name, model_config in base_models:
        if name == 'LSTM':
            # Load pre-trained LSTM model
            model = tf.keras.models.load_model('/root/gpu_model_default.h5')
            X_train_3d = tf.expand_dims(tf.convert_to_tensor(X_train.astype(np.float32)), axis=1)
            train_pred = model.predict(X_train_3d).flatten()
            # Calculate R² for LSTM on training set
            r2 = r2_score(y_train, train_pred)
            model_weights[name] = max(r2, 0.0)  # Ensure non-negative weights
            # Calculate entropy (using prediction variance as proxy)
            variance = np.var(train_pred)
            model_weights[name] /= (1 + variance)  # Adjust weights by entropy
        else:
            # For Ridge and XGBoost, use K-fold cross-validation
            r2_scores = []
            for train_idx, valid_idx in kf.split(X_train):
                X_tr, X_val = X_train[train_idx], X_train[valid_idx]
                y_tr, y_val = y_train[train_idx], y_train[valid_idx]
                model = model_config['model'](**model_config['params'])
                if name == 'Ridge':
                    model.fit(cp.asarray(X_tr), cp.asarray(y_tr))
                    pred_val = model.predict(cp.asarray(X_val))
                    if isinstance(pred_val, cp.ndarray):
                        pred_val = pred_val.get()
                else:
                    model.fit(X_tr, y_tr)
                    pred_val = model.predict(X_val)
                r2 = r2_score(y_val, pred_val)
                r2_scores.append(r2)
            avg_r2 = np.mean(r2_scores)
            model_weights[name] = max(avg_r2, 0.0)  # Ensure non-negative weights
            # Calculate entropy (using prediction variance as proxy)
            model = model_config['model'](**model_config['params'])
            if name == 'Ridge':
                model.fit(cp.asarray(X_train), cp.asarray(y_train))
                pred_train = model.predict(cp.asarray(X_train))
                if isinstance(pred_train, cp.ndarray):
                    train_pred = pred_train.get()
                else:
                    train_pred = pred_train
            else:  # XGBoost
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
            variance = np.var(train_pred)
            model_weights[name] /= (1 + variance)  # Adjust weights by entropy

    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight == 0:
        # If all weights are 0, assign uniform weights
        for name in model_weights:
            model_weights[name] = 1.0 / len(model_weights)
    else:
        for name in model_weights:
            model_weights[name] /= total_weight

    # Generate weighted meta-features
    log_step("Generating weighted meta-features")
    base_preds_train_weighted = []
    base_preds_test_weighted = []

    for name, model_config in base_models:
        if name == 'LSTM':
            model = tf.keras.models.load_model('/root/gpu_model_default.h5')
            X_train_3d = tf.expand_dims(tf.convert_to_tensor(X_train.astype(np.float32)), axis=1)
            X_test_3d = tf.expand_dims(tf.convert_to_tensor(X_test.astype(np.float32)), axis=1)
            train_pred = model.predict(X_train_3d).flatten()
            test_pred = model.predict(X_test_3d).flatten()
        else:
            model = model_config['model'](**model_config['params'])
            if name == 'Ridge':
                model.fit(cp.asarray(X_train), cp.asarray(y_train))
                pred_train = model.predict(cp.asarray(X_train))
                pred_test = model.predict(cp.asarray(X_test))
                if isinstance(pred_train, cp.ndarray):
                    train_pred = pred_train.get()
                else:
                    train_pred = pred_train
                if isinstance(pred_test, cp.ndarray):
                    test_pred = pred_test.get()
                else:
                    test_pred = pred_test
            else:  # XGBoost
                model.fit(X_train, y_train)
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
        # Weighted predictions
        weight = model_weights[name]
        base_preds_train_weighted.append(train_pred.reshape(-1, 1) * weight)
        base_preds_test_weighted.append(test_pred.reshape(-1, 1) * weight)

    # Combine weighted predictions
    meta_X_train = np.hstack(base_preds_train_weighted)
    meta_X_test = np.hstack(base_preds_test_weighted)

    # Train meta-model
    log_step("Training meta-model")
    meta_model = Ridge(alpha=0.5)
    meta_model.fit(meta_X_train.astype(np.float32), y_train.astype(np.float32))

    # Generate final predictions on test set
    final_pred = meta_model.predict(meta_X_test.astype(np.float32))

    # Evaluate model
    metrics = evaluate_model(y_test, final_pred)

    # Save predictions and true values to CSV
    pd.DataFrame({'True': y_test, 'Pred': final_pred}).to_csv('predictions.csv', index=False)
    log_step("Predictions and true values saved to 'predictions.csv'")

    return metrics

# Example run
if __name__ == "__main__":
    config = {
        "file_path": "/root/database.xlsx",
        "features": ['ir', 'chargetime', 'Qdlin', 'SOH', 'Voltage_m', 'Current_m', 'Temp_m', 'Current_l', 'Voltage_l', 'CCCT', 'CVCT'],
        "target": 'QDischarge'
    }
    results = stacking_workflow(**config)
    print("\nFinal evaluation metrics:")
    print(f"- RMSE: {results['RMSE']:.4f}\n- MAE: {results['MAE']:.4f}\n- R²: {results['R2']:.4f}")