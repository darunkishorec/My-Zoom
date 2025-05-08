"""
Training script for the My Zoom feedback classification model.
"""

import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from data_preprocessing import DataPreprocessor
from model import ZoomFeedbackClassifier, ModelTrainer

def plot_training_history(history, save_path=None):
    """
    Plot the training history and optionally save the figure.
    
    Args:
        history (dict): Training history
        save_path (str): Path to save the figure (optional)
    """
    # Create figure with 2 rows and 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss')
    ax1.plot(history['eval_loss'], label='Validation Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['accuracy'], label='Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    
    # Plot precision and recall
    ax3.plot(history['precision'], label='Precision')
    ax3.plot(history['recall'], label='Recall')
    ax3.set_title('Precision and Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.legend()
    
    # Plot F1 score
    ax4.plot(history['f1'], label='F1 Score')
    ax4.set_title('F1 Score')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('F1')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Create timestamp for model files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    preprocessor = DataPreprocessor(max_length=args.max_seq_length)
    
    train_encodings, eval_encodings, train_df, eval_df = preprocessor.prepare_data_for_training(
        args.train_data,
        args.eval_data,
        clean=True,
        remove_stop=args.remove_stopwords,
        augment=args.augment_data
    )
    
    # Print data statistics
    print(f"Training samples: {len(train_df)}")
    print(f"Evaluation samples: {len(eval_df)}")
    print(f"Class distribution in training: {train_df['label'].value_counts().to_dict()}")
    
    # Initialize model
    print(f"Initializing model with {args.model_name}...")
    model = ZoomFeedbackClassifier(
        pretrained_model_name=args.model_name,
        dropout_prob=args.dropout
    )
    
    # Initialize trainer
    trainer = ModelTrainer(model, device, lr=args.learning_rate)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = trainer.create_dataloaders(
        train_encodings, 
        eval_encodings,
        batch_size=args.batch_size
    )
    
    # Train model
    print("Starting training...")
    model_path = os.path.join(args.output_dir, f"zoom_feedback_model_{timestamp}.pt")
    
    history = trainer.train(
        train_dataloader,
        eval_dataloader,
        epochs=args.epochs,
        save_path=model_path
    )
    
    # Plot and save training history
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, f"training_history_{timestamp}.png")
        plot_training_history(history, plot_path)
    
    # Save model config
    config_path = os.path.join(args.output_dir, f"model_config_{timestamp}.txt")
    with open(config_path, 'w') as f:
        f.write(f"Model name: {args.model_name}\n")
        f.write(f"Max sequence length: {args.max_seq_length}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Augmented data: {args.augment_data}\n")
        f.write(f"Removed stopwords: {args.remove_stopwords}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Final metrics:\n")
        f.write(f"  Train loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Eval loss: {history['eval_loss'][-1]:.4f}\n")
        f.write(f"  Accuracy: {history['accuracy'][-1]:.4f}\n")
        f.write(f"  Precision: {history['precision'][-1]:.4f}\n")
        f.write(f"  Recall: {history['recall'][-1]:.4f}\n")
        f.write(f"  F1 Score: {history['f1'][-1]:.4f}\n")
    
    print(f"Training completed. Model saved to {model_path}")
    print(f"Model configuration saved to {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT-based model for Zoom feedback validation")
    
    parser.add_argument("--train_data", type=str, required=False, default="../train.xlsx",
                        help="Path to training data")
    parser.add_argument("--eval_data", type=str, required=False, default="../evaluation.xlsx",
                        help="Path to evaluation data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the pretrained model to use")
    parser.add_argument("--output_dir", type=str, default="../models",
                        help="Output directory for model and artifacts")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--augment_data", action="store_true",
                        help="Augment minority class data")
    parser.add_argument("--remove_stopwords", action="store_true",
                        help="Remove stopwords from text")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plotting training history")
    
    args = parser.parse_args()
    
    main(args)
