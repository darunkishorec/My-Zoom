"""
Evaluation script for the My Zoom feedback classification model.
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, precision_recall_curve, auc
)
from data_preprocessing import DataPreprocessor
from model import ZoomFeedbackClassifier, FeedbackDataset, ModelTrainer
from torch.utils.data import DataLoader

def plot_confusion_matrix(confusion_mat, class_names, save_path=None):
    """
    Plot the confusion matrix.
    
    Args:
        confusion_mat (ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str): Path to save the plot (optional)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mat, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plot the ROC curve.
    
    Args:
        y_true (list): True labels
        y_prob (list): Predicted probabilities for the positive class
        save_path (str): Path to save the plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Plot the Precision-Recall curve.
    
    Args:
        y_true (list): True labels
        y_prob (list): Predicted probabilities for the positive class
        save_path (str): Path to save the plot (optional)
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()
    
    return pr_auc

def analyze_misclassifications(eval_df, y_pred, output_path=None):
    """
    Analyze and save misclassified examples.
    
    Args:
        eval_df (DataFrame): Evaluation dataframe
        y_pred (list): Model predictions
        output_path (str): Path to save the analysis (optional)
        
    Returns:
        DataFrame: Misclassified examples
    """
    # Add predictions to dataframe
    eval_df = eval_df.copy()
    eval_df['prediction'] = y_pred
    
    # Find misclassified examples
    misclassified = eval_df[eval_df['label'] != eval_df['prediction']]
    
    if output_path:
        misclassified.to_csv(output_path, index=False)
        print(f"Misclassified examples saved to {output_path}")
    
    return misclassified

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load and preprocess evaluation data
    print("Loading and preprocessing evaluation data...")
    preprocessor = DataPreprocessor(max_length=args.max_seq_length)
    
    _, eval_df = preprocessor.load_data(args.train_data, args.eval_data)
    eval_df = preprocessor.preprocess_data(eval_df, clean=True, remove_stop=args.remove_stopwords)
    
    # Tokenize data
    eval_encodings = preprocessor.tokenize_data(
        eval_df['cleaned_text'].tolist(),
        eval_df['label'].tolist()
    )
    
    # Create dataset and dataloader
    eval_dataset = FeedbackDataset(eval_encodings)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ZoomFeedbackClassifier(
        pretrained_model_name=args.model_name,
        dropout_prob=0.1  # Dropout doesn't matter for evaluation
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Evaluate model
    print("Evaluating model...")
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get predictions
            preds = torch.argmax(outputs, dim=1)
            
            all_labels.extend(batch['labels'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Prob of positive class
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['Invalid', 'Valid'])
    print("\nClassification Report:")
    print(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results to file
    results_path = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("=================\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Evaluation data: {args.eval_data}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"Evaluation results saved to {results_path}")
    
    # Plot confusion matrix
    if not args.no_plot:
        cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
        plot_confusion_matrix(cm, ['Invalid', 'Valid'], cm_path)
        
        # Plot ROC curve
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        roc_auc = plot_roc_curve(all_labels, all_probs, roc_path)
        
        # Plot Precision-Recall curve
        pr_path = os.path.join(args.output_dir, "pr_curve.png")
        pr_auc = plot_precision_recall_curve(all_labels, all_probs, pr_path)
        
        # Update results file with AUC values
        with open(results_path, 'a') as f:
            f.write(f"\n\nROC AUC: {roc_auc:.3f}")
            f.write(f"\nPR AUC: {pr_auc:.3f}")
    
    # Analyze misclassifications
    if args.analyze_errors:
        misclass_path = os.path.join(args.output_dir, "misclassified_examples.csv")
        misclassified = analyze_misclassifications(eval_df, all_preds, misclass_path)
        print(f"\nNumber of misclassified examples: {len(misclassified)}")
        
        # Display a few misclassified examples
        if len(misclassified) > 0:
            print("\nSample of misclassified examples:")
            for i, (_, row) in enumerate(misclassified.head(5).iterrows()):
                print(f"Example {i+1}:")
                print(f"  Text: {row['text']}")
                print(f"  True label: {row['label']}")
                print(f"  Predicted: {row['prediction']}")
                if 'reason' in row and isinstance(row['reason'], str) and len(row['reason']) > 0:
                    print(f"  Reason: {row['reason']}")
                print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Zoom feedback classification model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--train_data", type=str, required=False, default="../train.xlsx",
                        help="Path to training data (for reference)")
    parser.add_argument("--eval_data", type=str, required=False, default="../evaluation.xlsx",
                        help="Path to evaluation data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the pretrained model used for training")
    parser.add_argument("--output_dir", type=str, default="../evaluation",
                        help="Output directory for evaluation results")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="Remove stopwords from text (should match training)")
    parser.add_argument("--analyze_errors", action="store_true",
                        help="Analyze misclassified examples")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plotting")
    
    args = parser.parse_args()
    
    main(args)
