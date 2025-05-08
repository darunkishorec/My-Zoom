"""
Utility functions for the My Zoom feedback classification project.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import seaborn as sns

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def get_device():
    """
    Get the device to use for computation.
    
    Returns:
        torch.device: Device (CPU or GPU)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_directory(directory_path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if created or already exists, False otherwise
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {str(e)}")
        return False
        
def load_model(model_class, model_path, device=None, **model_kwargs):
    """
    Load a trained model from a file.
    
    Args:
        model_class: Model class to instantiate
        model_path (str): Path to the saved model
        device (torch.device): Device to load the model to
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        model: Loaded model
    """
    if device is None:
        device = get_device()
        
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def get_class_distribution(labels):
    """
    Get the distribution of classes in a dataset.
    
    Args:
        labels (list or ndarray): Class labels
        
    Returns:
        dict: Class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))

def plot_metrics(history, output_path=None):
    """
    Plot training metrics.
    
    Args:
        history (dict): Training history dictionary
        output_path (str): Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['eval_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['accuracy'])
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    
    # Precision and Recall
    axes[1, 0].plot(history['precision'], label='Precision')
    axes[1, 0].plot(history['recall'], label='Recall')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    
    # F1 Score
    axes[1, 1].plot(history['f1'])
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names=None, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (list or ndarray): True labels
        y_pred (list or ndarray): Predicted labels
        class_names (list): List of class names
        output_path (str): Path to save the plot (optional)
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(cm.shape[0])]
    
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        plt.savefig(output_path)
        
    return fig

def plot_roc_curve(y_true, y_score, output_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true (list or ndarray): True labels
        y_score (list or ndarray): Predicted probabilities
        output_path (str): Path to save the plot (optional)
        
    Returns:
        tuple: (fig, roc_auc)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if output_path:
        plt.savefig(output_path)
        
    return fig, roc_auc

def plot_pr_curve(y_true, y_score, output_path=None):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true (list or ndarray): True labels
        y_score (list or ndarray): Predicted probabilities
        output_path (str): Path to save the plot (optional)
        
    Returns:
        tuple: (fig, pr_auc)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    fig = plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    if output_path:
        plt.savefig(output_path)
        
    return fig, pr_auc

def save_misclassified_examples(df, y_true, y_pred, output_path):
    """
    Save misclassified examples to a CSV file.
    
    Args:
        df (DataFrame): Original dataframe
        y_true (list or ndarray): True labels
        y_pred (list or ndarray): Predicted labels
        output_path (str): Path to save the CSV file
        
    Returns:
        DataFrame: Misclassified examples
    """
    df_copy = df.copy()
    df_copy['true_label'] = y_true
    df_copy['predicted_label'] = y_pred
    
    misclassified = df_copy[df_copy['true_label'] != df_copy['predicted_label']]
    
    if output_path:
        misclassified.to_csv(output_path, index=False)
        
    return misclassified

def preprocess_input_text(text, preprocessor=None, tokenizer=None, max_length=128):
    """
    Preprocess input text for inference.
    
    Args:
        text (str): Input text
        preprocessor: Preprocessor instance (optional)
        tokenizer: Tokenizer instance (optional)
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Preprocessed input for the model
    """
    if preprocessor:
        text = preprocessor.clean_text(text)
    
    if tokenizer:
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
    
    return text
