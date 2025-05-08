"""
Fine-tuning Script for the My Zoom Feedback Validation Model

This script allows users to fine-tune an existing model with new data,
enabling the model to adapt to changing feedback patterns over time.
"""

import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import BertTokenizer
from src.data_preprocessing import DataPreprocessor
from src.model import ZoomFeedbackClassifier, ModelTrainer, FeedbackDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

def plot_training_history(history, output_path=None):
    """Plot training history metrics and save to file if requested."""
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
    
    if output_path:
        plt.savefig(output_path)
        print(f"Training history plot saved to {output_path}")
    
    plt.show()

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create timestamp for model files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize preprocessor and tokenizer
    print("Initializing preprocessor and tokenizer...")
    preprocessor = DataPreprocessor(max_length=args.max_seq_length, tokenizer_name=args.model_name)
    
    # Load new data
    print(f"Loading new data from {args.new_data}...")
    
    # Determine file type based on extension
    file_ext = os.path.splitext(args.new_data)[1].lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(args.new_data)
    elif file_ext == '.csv':
        df = pd.read_csv(args.new_data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Verify required columns exist
    required_columns = [args.text_column, args.label_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Preprocess new data
    print("Preprocessing new data...")
    df = preprocessor.preprocess_data(df, clean=True, remove_stop=args.remove_stopwords)
    
    # Split into train and validation sets
    train_df, val_df = train_test_split(
        df, 
        test_size=args.val_size,
        random_state=args.seed,
        stratify=df[args.label_column]
    )
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Display class distribution
    train_class_counts = train_df[args.label_column].value_counts()
    val_class_counts = val_df[args.label_column].value_counts()
    
    print("Class distribution in training set:")
    print(train_class_counts)
    print("Class distribution in validation set:")
    print(val_class_counts)
    
    # Augment if requested
    if args.augment_data:
        class_counts = train_df[args.label_column].value_counts()
        minority_class = class_counts.idxmin()
        augmentation_factor = (class_counts.max() // class_counts.min()) - 1
        
        if augmentation_factor > 0:
            print(f"Augmenting minority class {minority_class} by factor {augmentation_factor}...")
            train_df = preprocessor.augment_data(
                train_df, 
                minority_class=minority_class,
                augmentation_factor=augmentation_factor
            )
            print(f"After augmentation: {len(train_df)} training samples")
    
    # Tokenize data
    print("Tokenizing data...")
    train_encodings = preprocessor.tokenize_data(
        train_df['cleaned_text'].tolist(),
        train_df[args.label_column].tolist()
    )
    
    val_encodings = preprocessor.tokenize_data(
        val_df['cleaned_text'].tolist(),
        val_df[args.label_column].tolist()
    )
    
    # Load pretrained model if available
    print(f"Initializing model with {args.model_name}...")
    model = ZoomFeedbackClassifier(
        pretrained_model_name=args.model_name,
        dropout_prob=args.dropout
    )
    
    # Load existing model weights if specified
    if args.base_model_path and os.path.exists(args.base_model_path):
        print(f"Loading base model from {args.base_model_path}...")
        model.load_state_dict(torch.load(args.base_model_path, map_location=device))
    
    # Initialize trainer
    trainer = ModelTrainer(model, device, lr=args.learning_rate)
    
    # Create dataloaders
    train_dataloader, val_dataloader = trainer.create_dataloaders(
        train_encodings,
        val_encodings,
        batch_size=args.batch_size
    )
    
    # Fine-tune model
    print(f"Fine-tuning model for {args.epochs} epochs...")
    model_path = os.path.join(args.output_dir, f"zoom_feedback_model_finetuned_{timestamp}.pt")
    
    history = trainer.train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        save_path=model_path
    )
    
    # Plot and save training history
    if not args.no_plot:
        plot_path = os.path.join(args.output_dir, f"finetuning_history_{timestamp}.png")
        plot_training_history(history, plot_path)
    
    # Save model config
    config_path = os.path.join(args.output_dir, f"model_config_finetuned_{timestamp}.txt")
    with open(config_path, 'w') as f:
        f.write(f"Model name: {args.model_name}\n")
        f.write(f"Base model: {args.base_model_path}\n")
        f.write(f"New data: {args.new_data}\n")
        f.write(f"Max sequence length: {args.max_seq_length}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Dropout: {args.dropout}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Augmented data: {args.augment_data}\n")
        f.write(f"Removed stopwords: {args.remove_stopwords}\n")
        f.write(f"Validation size: {args.val_size}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Final metrics:\n")
        f.write(f"  Train loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Eval loss: {history['eval_loss'][-1]:.4f}\n")
        f.write(f"  Accuracy: {history['accuracy'][-1]:.4f}\n")
        f.write(f"  Precision: {history['precision'][-1]:.4f}\n")
        f.write(f"  Recall: {history['recall'][-1]:.4f}\n")
        f.write(f"  F1 Score: {history['f1'][-1]:.4f}\n")
    
    print(f"Fine-tuning completed. Model saved to {model_path}")
    print(f"Model configuration saved to {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the My Zoom feedback validation model with new data")
    
    parser.add_argument("--new_data", type=str, required=True,
                        help="Path to new data file (Excel or CSV)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing the feedback texts")
    parser.add_argument("--label_column", type=str, default="label",
                        help="Column name containing the labels (0=Invalid, 1=Valid)")
    parser.add_argument("--base_model_path", type=str,
                        help="Path to existing model to use as starting point")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the pretrained model")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="Proportion of data to use for validation")
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
