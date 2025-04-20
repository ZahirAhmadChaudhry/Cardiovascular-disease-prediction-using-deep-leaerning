#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image-based Cardiovascular Disease Prediction Model

This module implements a convolutional neural network (CNN) for predicting
cardiovascular disease based on medical heart scan images.

Author: Ahmad Zahir
Last Modified: April 20, 2025
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'image_model.log'))
    ]
)
logger = logging.getLogger('CardiovascularImageModel')

# Configuration constants
IMAGE_SIZE = 299  # Default image size (width & height)
BATCH_SIZE = 32   # Batch size for training
CHANNELS = 3      # Default images are in RGB (three channels)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Saved_model', 'img_mdl.h5')

class CardiovascularImageModel:
    """
    A CNN-based model for cardiovascular disease prediction from medical images.
    
    This class handles:
    - Data loading and preprocessing
    - Model architecture definition
    - Training and evaluation
    - Model saving
    """
    
    def __init__(self, dataset_path, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
        """
        Initialize the model with configuration parameters.
        
        Args:
            dataset_path (str): Path to the dataset directory
            img_size (int): Size of images for processing (square)
            batch_size (int): Batch size for training
        """
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.dataset = None
        self.class_names = None
        self.train_ds = None
        self.test_ds = None
        self.val_ds = None
        self.history = None
        
        # Data augmentation and preprocessing components
        self.resize_and_rescale = tf.keras.Sequential([
            layers.Resizing(self.img_size, self.img_size),
            layers.Rescaling(1.0/255)
        ])
        
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2)
        ])
        
    def load_dataset(self):
        """
        Load the dataset from the specified path.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            self.dataset = tf.keras.utils.image_dataset_from_directory(
                self.dataset_path,
                shuffle=True,
                image_size=(self.img_size, self.img_size),
                batch_size=self.batch_size
            )
            
            self.class_names = self.dataset.class_names
            logger.info(f"Dataset loaded successfully with classes: {self.class_names}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return False
            
    def visualize_sample_images(self, samples=12, rows=3, cols=4):
        """
        Visualize sample images from the dataset.
        
        Args:
            samples (int): Number of samples to display
            rows (int): Number of rows in the display grid
            cols (int): Number of columns in the display grid
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return
            
        try:
            plt.figure(figsize=(15, 12))
            for image_batch, label_batch in self.dataset.take(1):
                for i in range(min(samples, len(image_batch))):
                    plt.subplot(rows, cols, i+1)
                    plt.imshow(image_batch[i].numpy().astype("uint8"))
                    plt.title(self.class_names[label_batch[i]])
                    plt.axis("off")
            plt.tight_layout()
            plt.savefig('sample_images.png')
            plt.close()
            logger.info(f"Sample visualization saved to 'sample_images.png'")
        except Exception as e:
            logger.error(f"Error visualizing samples: {str(e)}")
            
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            train_ratio (float): Proportion of data for training
            val_ratio (float): Proportion of data for validation
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.dataset is None:
            logger.error("Dataset not loaded. Call load_dataset() first.")
            return False
            
        try:
            dataset_size = len(self.dataset)
            train_size = int(dataset_size * train_ratio)
            val_size = int(dataset_size * val_ratio)
            
            self.train_ds = self.dataset.take(train_size)
            remaining = self.dataset.skip(train_size)
            self.val_ds = remaining.take(val_size)
            self.test_ds = remaining.skip(val_size)
            
            # Optimize for performance
            self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.val_ds = self.val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            self.test_ds = self.test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            
            logger.info(f"Dataset split: {train_size} training batches, {val_size} validation batches, "
                       f"{len(self.test_ds)} test batches")
            return True
        except Exception as e:
            logger.error(f"Error splitting dataset: {str(e)}")
            return False
            
    def build_model(self, n_classes=2):
        """
        Build the CNN model architecture.
        
        Args:
            n_classes (int): Number of output classes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            input_shape = (CHANNELS, self.img_size, self.img_size, CHANNELS)
            
            self.model = models.Sequential([
                # Image preprocessing layers
                self.resize_and_rescale,
                self.data_augmentation,
                
                # Convolutional layers
                layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.MaxPooling2D((2, 2)),
                
                # Classification layers
                layers.Flatten(),
                layers.Dropout(0.5),  # Add dropout for regularization
                layers.Dense(128, activation='relu'),
                layers.Dense(n_classes, activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
            )
            
            self.model.build(input_shape=input_shape)
            self.model.summary()
            logger.info("Model built successfully")
            return True
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            return False
            
    def train(self, epochs=100):
        """
        Train the model with early stopping.
        
        Args:
            epochs (int): Maximum number of epochs
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.model is None:
            logger.error("Model not built. Call build_model() first.")
            return False
            
        if self.train_ds is None or self.val_ds is None:
            logger.error("Dataset not split. Call split_dataset() first.")
            return False
            
        try:
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=10,
                verbose=1,
                mode='auto',
                restore_best_weights=True
            )
            
            # Add TensorBoard logging
            log_dir = os.path.join('logs', 'fit', datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1, write_graph=True
            )
            
            # Model checkpoint to save best model
            checkpoint_path = os.path.join('checkpoints', 'model_checkpoint.h5')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
            
            # Train model
            logger.info(f"Starting training for {epochs} epochs")
            self.history = self.model.fit(
                self.train_ds,
                epochs=epochs,
                validation_data=self.val_ds,
                callbacks=[early_stopping, tensorboard_callback, model_checkpoint],
                verbose=1
            )
            
            logger.info("Training completed")
            return True
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
            
    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
            
        if self.test_ds is None:
            logger.error("Test dataset not available. Call split_dataset() first.")
            return None
            
        try:
            logger.info("Evaluating model on test dataset")
            scores = self.model.evaluate(self.test_ds, verbose=1)
            
            metrics = {
                'loss': scores[0],
                'accuracy': scores[1],
                'precision': scores[2] if len(scores) > 2 else None,
                'recall': scores[3] if len(scores) > 3 else None
            }
            
            logger.info(f"Evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            return None
            
    def plot_training_history(self):
        """
        Plot the training history (accuracy and loss).
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.history is None:
            logger.error("No training history available. Call train() first.")
            return False
            
        try:
            # Plot accuracy
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            plt.close()
            
            logger.info("Training history plots saved to 'training_history.png'")
            return True
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            return False
            
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
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False


def main():
    """
    Main function to run the cardiovascular disease prediction model training.
    """
    # Get dataset path from environment or use default
    dataset_path = os.environ.get('DATASET_PATH', '/path/to/your/image/dataset')
    
    # Initialize model
    model = CardiovascularImageModel(dataset_path)
    
    # Check if we should run training or just load a pre-trained model
    should_train = os.environ.get('TRAIN_MODEL', 'False').lower() == 'true'
    
    if should_train:
        # Training workflow
        if model.load_dataset():
            model.visualize_sample_images()
            
            if model.split_dataset():
                if model.build_model():
                    if model.train():
                        model.evaluate()
                        model.plot_training_history()
                        model.save_model()
    else:
        logger.info("Skipping training. Set TRAIN_MODEL=True to enable training.")


if __name__ == "__main__":
    main()