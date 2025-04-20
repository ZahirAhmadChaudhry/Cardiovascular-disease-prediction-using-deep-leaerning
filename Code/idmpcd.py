#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tabular Data-based Cardiovascular Disease Prediction Model

This module implements a deep learning model for predicting
cardiovascular disease based on patient clinical features.

Author: Ahmad Zahir
Last Modified: April 20, 2025
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'tabular_model.log'))
    ]
)
logger = logging.getLogger('CardiovascularTabularModel')

# Configuration constants
RANDOM_SEED = 42
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_model', 'tb_mdl.h5')
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'Tabular_data.csv')

class CardiovascularTabularModel:
    """
    A deep learning model for cardiovascular disease prediction based on tabular patient data.
    
    This class handles:
    - Data loading and preprocessing
    - Feature engineering
    - Model architecture definition
    - Training with cross-validation
    - Evaluation with multiple metrics
    - Model interpretation and visualization
    - Model saving
    """
    
    def __init__(self, dataset_path=DEFAULT_DATASET_PATH):
        """
        Initialize the model with configuration parameters.
        
        Args:
            dataset_path (str): Path to the dataset CSV file
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_names = None
        
        # Set random seeds for reproducibility
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
    def load_data(self):
        """
        Load and perform initial processing of the data.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading data from: {self.dataset_path}")
            self.data = pd.read_csv(self.dataset_path)
            logger.info(f"Data loaded successfully with shape: {self.data.shape}")
            
            # Basic data information
            logger.info("Dataset information:")
            logger.info(f"Columns: {self.data.columns.tolist()}")
            logger.info(f"Data types: {self.data.dtypes}")
            logger.info(f"Missing values: {self.data.isnull().sum().sum()}")
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
    
    def explore_data(self):
        """
        Perform exploratory data analysis and generate visualizations.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
            
        try:
            # Create output directory for visualizations
            viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Data statistics
            stats_df = pd.DataFrame({
                'Mean': self.data.mean(),
                'Median': self.data.median(),
                'Std': self.data.std(),
                'Min': self.data.min(),
                'Max': self.data.max(),
                'Skew': self.data.skew(),
                'Kurtosis': self.data.kurtosis()
            })
            stats_df.to_csv(os.path.join(viz_dir, 'data_statistics.csv'))
            
            # Target distribution
            if 'target' in self.data.columns:
                plt.figure(figsize=(10, 6))
                sns.countplot(x='target', data=self.data)
                plt.title('Target Distribution')
                plt.savefig(os.path.join(viz_dir, 'target_distribution.png'))
                plt.close()
                
                # Check class balance
                target_counts = self.data['target'].value_counts()
                logger.info(f"Target distribution: {target_counts.to_dict()}")
                
                # Feature correlations
                plt.figure(figsize=(12, 10))
                corr_matrix = self.data.corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
                plt.title('Feature Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'correlation_matrix.png'))
                plt.close()
                
                # Correlations with target
                target_corr = corr_matrix['target'].sort_values(ascending=False)
                logger.info(f"Feature correlations with target:\n{target_corr}")
                
                # Histograms for numerical features
                self.data.hist(figsize=(15, 12), bins=20)
                plt.suptitle('Feature Distributions')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'feature_distributions.png'))
                plt.close()
            
            return True
        except Exception as e:
            logger.error(f"Error in exploratory data analysis: {str(e)}")
            return False
    
    def preprocess_data(self, test_size=0.2, scale_method='standard'):
        """
        Preprocess the data: handle missing values, outliers, and split into train/test sets.
        
        Args:
            test_size (float): Proportion of data for testing
            scale_method (str): Scaling method: 'standard' or 'minmax'
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.data is None:
            logger.error("Data not loaded. Call load_data() first.")
            return False
            
        try:
            # Identify feature columns and target
            if 'target' in self.data.columns:
                target_col = 'target'
            else:
                # Try to find a column that looks like a target
                for col in self.data.columns:
                    if 'target' in col.lower() or 'label' in col.lower() or 'class' in col.lower():
                        target_col = col
                        break
                else:
                    # If no target-like column is found, assume the last column is the target
                    target_col = self.data.columns[-1]
            
            logger.info(f"Using '{target_col}' as the target column")
            
            # Convert target to binary (0 or 1)
            # Any value > 0 is considered a positive class (1)
            self.data[target_col] = (self.data[target_col] > 0).astype(int)
            logger.info(f"Target converted to binary. New distribution: {self.data[target_col].value_counts().to_dict()}")
            
            # Separate features and target
            X = self.data.drop(columns=[target_col])
            y = self.data[target_col]
            self.feature_names = X.columns.tolist()
            
            # Handle any missing values
            if X.isnull().sum().sum() > 0:
                logger.info("Handling missing values")
                # For numerical columns, fill with mean
                num_cols = X.select_dtypes(include=['float64', 'int64']).columns
                for col in num_cols:
                    if X[col].isnull().sum() > 0:
                        X[col] = X[col].fillna(X[col].mean())
                
                # For categorical columns, fill with mode
                cat_cols = X.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    if X[col].isnull().sum() > 0:
                        X[col] = X[col].fillna(X[col].mode()[0])
            
            # Handle categorical features if any
            cat_cols = X.select_dtypes(include=['object', 'category']).columns
            if len(cat_cols) > 0:
                logger.info(f"One-hot encoding categorical features: {cat_cols.tolist()}")
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
                self.feature_names = X.columns.tolist()
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
            )
            
            logger.info(f"Data split: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
            
            # Scale features
            if scale_method == 'standard':
                self.scaler = StandardScaler()
                logger.info("Using StandardScaler for feature scaling")
            else:  # minmax
                self.scaler = MinMaxScaler()
                logger.info("Using MinMaxScaler for feature scaling")
                
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            # Save processed data
            self.X = X
            self.y = y
            
            return True
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            return False
    
    def build_model(self, hidden_layers=None, dropout_rate=0.3, regularization=0.001):
        """
        Build the neural network model.
        
        Args:
            hidden_layers (list): List of neurons in each hidden layer
            dropout_rate (float): Dropout rate for regularization
            regularization (float): L1/L2 regularization factor
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.X_train is None:
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            return False
            
        try:
            # Default architecture if none provided
            if hidden_layers is None:
                input_dim = self.X_train.shape[1]
                hidden_layers = [
                    input_dim * 2,
                    input_dim,
                    max(8, input_dim // 2)
                ]
            
            logger.info(f"Building model with hidden layers: {hidden_layers}")
            
            # Create model
            model = Sequential()
            
            # Input layer
            model.add(Dense(
                hidden_layers[0],
                input_dim=self.X_train.shape[1],
                activation='relu',
                kernel_regularizer=l1_l2(l1=regularization, l2=regularization)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
            
            # Hidden layers
            for units in hidden_layers[1:]:
                model.add(Dense(
                    units,
                    activation='relu',
                    kernel_regularizer=l1_l2(l1=regularization, l2=regularization)
                ))
                model.add(BatchNormalization())
                model.add(Dropout(dropout_rate))
            
            # Output layer
            model.add(Dense(1, activation='sigmoid'))
            
            # Compile model with Adam optimizer
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.AUC()
                ]
            )
            
            model.summary()
            self.model = model
            
            return True
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False
    
    def train(self, epochs=1000, batch_size=32, validation_split=0.2, patience=20):
        """
        Train the model with early stopping and learning rate reduction.
        
        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size for training
            validation_split (float): Proportion of training data for validation
            patience (int): Patience for early stopping
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return False
            
        try:
            # Create directories for logs and checkpoints
            log_dir = os.path.join('logs', 'fit', datetime.now().strftime("%Y%m%d-%H%M%S"))
            checkpoint_dir = os.path.join('checkpoints')
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Callbacks
            callbacks = [
                # Early stopping to prevent overfitting
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                # Reduce learning rate when plateauing
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=patience // 2,
                    min_lr=1e-6,
                    verbose=1
                ),
                # Save best model during training
                ModelCheckpoint(
                    filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                # TensorBoard logging
                TensorBoard(log_dir=log_dir, histogram_freq=1)
            ]
            
            logger.info(f"Starting training for up to {epochs} epochs with batch size {batch_size}")
            start_time = time.time()
            
            # Train model
            self.history = self.model.fit(
                self.X_train,
                self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            return True
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
    
    def evaluate_model(self):
        """
        Evaluate the model on the test set with multiple metrics.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
            
        try:
            # Create output directory for evaluation results
            eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation')
            os.makedirs(eval_dir, exist_ok=True)
            
            # Predict on test set
            y_pred_prob = self.model.predict(self.X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred)
            }
            
            # Log metrics
            logger.info("Model evaluation on test set:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
            # Generate classification report
            class_report = classification_report(self.y_test, y_pred)
            logger.info(f"Classification Report:\n{class_report}")
            
            # Save classification report
            with open(os.path.join(eval_dir, 'classification_report.txt'), 'w') as f:
                f.write(class_report)
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease']
            )
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'))
            plt.close()
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)
            metrics['auc'] = roc_auc
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(eval_dir, 'roc_curve.png'))
            plt.close()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            return None
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.history is None:
            logger.error("No training history available. Call train() first.")
            return False
            
        try:
            # Create output directory
            viz_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Plot training & validation accuracy values
            plt.figure(figsize=(12, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'training_history.png'))
            plt.close()
            
            logger.info("Training history plots saved")
            return True
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            return False
    
    def cross_validate(self, n_splits=5, epochs=100, batch_size=32):
        """
        Perform k-fold cross-validation.
        
        Args:
            n_splits (int): Number of folds
            epochs (int): Maximum epochs per fold
            batch_size (int): Batch size
            
        Returns:
            dict: Cross-validation results
        """
        if self.X is None or self.y is None:
            logger.error("Data not preprocessed. Call preprocess_data() first.")
            return None
        
        try:
            # Initialize KFold
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
            
            # Metrics to track
            cv_scores = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': [],
                'auc': []
            }
            
            logger.info(f"Starting {n_splits}-fold cross-validation")
            
            # K-fold Cross Validation
            fold = 1
            for train_idx, val_idx in kfold.split(self.X):
                logger.info(f"Training fold {fold}/{n_splits}")
                
                # Split data
                X_train_fold, X_val_fold = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train_fold, y_val_fold = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                # Scale data
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_val_fold = scaler.transform(X_val_fold)
                
                # Build model
                model = Sequential()
                
                # Input layer
                model.add(Dense(units=self.X.shape[1]*2, activation='relu', input_dim=self.X.shape[1]))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
                # Hidden layers
                model.add(Dense(units=self.X.shape[1], activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
                model.add(Dense(units=max(8, self.X.shape[1]//2), activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(0.3))
                
                # Output layer
                model.add(Dense(1, activation='sigmoid'))
                
                # Compile model
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Early stopping
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=0
                )
                
                # Train model
                model.fit(
                    X_train_fold,
                    y_train_fold,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val_fold, y_val_fold),
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Evaluate
                y_pred_prob = model.predict(X_val_fold)
                y_pred = (y_pred_prob > 0.5).astype(int)
                
                # Calculate metrics
                acc = accuracy_score(y_val_fold, y_pred)
                prec = precision_score(y_val_fold, y_pred)
                rec = recall_score(y_val_fold, y_pred)
                f1 = f1_score(y_val_fold, y_pred)
                
                # ROC AUC
                fpr, tpr, _ = roc_curve(y_val_fold, y_pred_prob)
                roc_auc = auc(fpr, tpr)
                
                # Store scores
                cv_scores['accuracy'].append(acc)
                cv_scores['precision'].append(prec)
                cv_scores['recall'].append(rec)
                cv_scores['f1'].append(f1)
                cv_scores['auc'].append(roc_auc)
                
                logger.info(f"Fold {fold} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, "
                           f"Recall: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
                
                fold += 1
            
            # Calculate mean and std for all metrics
            cv_summary = {}
            for metric, scores in cv_scores.items():
                cv_summary[f'{metric}_mean'] = np.mean(scores)
                cv_summary[f'{metric}_std'] = np.std(scores)
            
            logger.info("Cross-validation results:")
            for metric, value in cv_summary.items():
                logger.info(f"{metric}: {value:.4f}")
                
            return cv_summary
            
        except Exception as e:
            logger.error(f"Error during cross-validation: {str(e)}")
            return None
    
    def save_model(self, filepath=None):
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model (defaults to MODEL_SAVE_PATH)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save. Call build_model() and train() first.")
            return False
            
        try:
            save_path = filepath if filepath else MODEL_SAVE_PATH
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            
            # Also save scaler for future preprocessing
            if self.scaler is not None:
                import joblib
                scaler_path = os.path.join(os.path.dirname(save_path), 'scaler.joblib')
                joblib.dump(self.scaler, scaler_path)
                logger.info(f"Scaler saved to {scaler_path}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False


def main():
    """
    Main function to run the cardiovascular disease prediction model training.
    """
    # Get dataset path from environment or use default
    dataset_path = os.environ.get('DATASET_PATH', DEFAULT_DATASET_PATH)
    
    # Initialize model
    model = CardiovascularTabularModel(dataset_path)
    
    # Check if we should run training or just load a pre-trained model
    should_train = os.environ.get('TRAIN_MODEL', 'True').lower() == 'true'
    
    if should_train:
        # Full workflow
        if model.load_data():
            model.explore_data()
            
            if model.preprocess_data():
                # Optional: Run cross-validation
                run_cv = os.environ.get('RUN_CV', 'False').lower() == 'true'
                if run_cv:
                    model.cross_validate()
                
                if model.build_model():
                    if model.train():
                        model.evaluate_model()
                        model.plot_training_history()
                        model.save_model()
    else:
        logger.info("Skipping training. Set TRAIN_MODEL=True to enable training.")


if __name__ == "__main__":
    main()