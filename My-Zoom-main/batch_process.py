"""
Batch Processing Script for the My Zoom Feedback Validation Project

This script processes multiple feedback texts in batch mode from an Excel file,
makes predictions using the trained model, and saves the results to a new Excel file.
"""

import os
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from src.model import ZoomFeedbackClassifier
from src.data_preprocessing import DataPreprocessor

def batch_predict(texts, model, preprocessor, tokenizer, device, batch_size=32):
    """
    Process a batch of texts and make predictions.
    
    Args:
        texts (list): List of input texts
        model: Trained model
        preprocessor: Data preprocessor
        tokenizer: BERT tokenizer
        device: Computation device
        batch_size (int): Batch size for processing
        
    Returns:
        dict: Prediction results with predictions, probabilities, and processed texts
    """
    # Preprocess texts
    processed_texts = [preprocessor.clean_text(text) for text in texts]
    
    # Prepare results containers
    all_predictions = []
    all_probs = []
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i+batch_size]
            
            # Tokenize batch
            encodings = tokenizer(
                batch_texts,
                add_special_tokens=True,
                max_length=preprocessor.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            # Move to device
            encodings = {k: v.to(device) for k, v in encodings.items()}
            
            # Get predictions
            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"]
            )
            
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get prediction labels
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_probs.extend(probs.cpu().numpy())
    
    # Prepare results
    results = {
        "processed_texts": processed_texts,
        "predictions": all_predictions,
        "probabilities": all_probs
    }
    
    return results

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Initialize preprocessor and tokenizer
    preprocessor = DataPreprocessor(max_length=args.max_seq_length)
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = ZoomFeedbackClassifier(
        pretrained_model_name=args.model_name,
        dropout_prob=0.1  # Dropout doesn't matter for inference
    )
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Make sure you've trained the model first using 'python src/train.py'")
        return
    
    # Load input data
    try:
        print(f"Loading data from {args.input_file}...")
        
        # Determine file type based on extension
        file_ext = os.path.splitext(args.input_file)[1].lower()
        
        if file_ext == '.xlsx' or file_ext == '.xls':
            df = pd.read_excel(args.input_file)
        elif file_ext == '.csv':
            df = pd.read_csv(args.input_file)
        elif file_ext == '.json':
            df = pd.read_json(args.input_file)
        else:
            print(f"Unsupported file format: {file_ext}")
            return
        
        # Verify text column exists
        if args.text_column not in df.columns:
            print(f"Text column '{args.text_column}' not found in input file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        texts = df[args.text_column].tolist()
        print(f"Loaded {len(texts)} texts for processing")
        
        # Process texts
        print("Processing texts...")
        results = batch_predict(
            texts, 
            model, 
            preprocessor, 
            tokenizer, 
            device, 
            batch_size=args.batch_size
        )
        
        # Add results to dataframe
        df['processed_text'] = results['processed_texts']
        df['prediction'] = results['predictions']
        df['prediction_label'] = df['prediction'].apply(lambda x: 'Valid' if x == 1 else 'Invalid')
        df['confidence'] = [probs[pred] for pred, probs in zip(results['predictions'], results['probabilities'])]
        df['valid_probability'] = [probs[1] for probs in results['probabilities']]
        df['invalid_probability'] = [probs[0] for probs in results['probabilities']]
        
        # Save results
        print(f"Saving results to {args.output_file}...")
        
        # Determine output file type based on extension
        out_ext = os.path.splitext(args.output_file)[1].lower()
        
        if out_ext == '.xlsx':
            df.to_excel(args.output_file, index=False)
        elif out_ext == '.csv':
            df.to_csv(args.output_file, index=False)
        elif out_ext == '.json':
            df.to_json(args.output_file, orient='records', indent=2)
        else:
            # Default to Excel
            df.to_excel(args.output_file, index=False)
        
        # Print summary
        valid_count = np.sum(df['prediction'] == 1)
        invalid_count = np.sum(df['prediction'] == 0)
        
        print("\nProcessing complete!")
        print(f"Total texts processed: {len(texts)}")
        print(f"Valid feedback: {valid_count} ({valid_count/len(texts)*100:.1f}%)")
        print(f"Invalid feedback: {invalid_count} ({invalid_count/len(texts)*100:.1f}%)")
        print(f"Results saved to {args.output_file}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process feedback texts with the My Zoom validation model")
    
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to input file (Excel, CSV, or JSON)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save results (Excel, CSV, or JSON)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing the feedback texts")
    parser.add_argument("--model_path", type=str, default="models/zoom_feedback_model.pt",
                        help="Path to the trained model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the pretrained model used for training")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    main(args)
