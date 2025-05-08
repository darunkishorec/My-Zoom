"""
Prediction script for the My Zoom feedback validation model.
This script allows users to make predictions on new feedback text using a trained model.
"""

import os
import torch
import argparse
from transformers import BertTokenizer
from src.model import ZoomFeedbackClassifier
from src.data_preprocessing import DataPreprocessor

def predict_feedback(text, model, preprocessor, tokenizer, device):
    """
    Predict whether the feedback text is valid or invalid.
    
    Args:
        text (str): Input feedback text
        model: Trained model
        preprocessor: Data preprocessor
        tokenizer: BERT tokenizer
        device: Computation device
        
    Returns:
        tuple: (prediction, probabilities, processed_text)
    """
    # Preprocess the text
    processed_text = preprocessor.clean_text(text)
    
    # Tokenize the text
    encoding = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=preprocessor.max_length,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    # Move to device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )
        
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        prediction = torch.argmax(probs, dim=1).item()
    
    return prediction, probs[0].cpu().numpy(), processed_text

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
    
    if args.input_file:
        # Process feedback from file
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"\nProcessing {len(texts)} feedback texts from file...")
            results = []
            
            for i, text in enumerate(texts):
                prediction, probs, processed_text = predict_feedback(
                    text, model, preprocessor, tokenizer, device
                )
                
                result = {
                    "original_text": text,
                    "processed_text": processed_text,
                    "prediction": "Valid" if prediction == 1 else "Invalid",
                    "confidence": float(probs[prediction]),
                    "valid_prob": float(probs[1]),
                    "invalid_prob": float(probs[0])
                }
                
                results.append(result)
                
                # Print results
                print(f"\nFeedback {i+1}:")
                print(f"Text: {text}")
                print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
                
            # Save results to file if output path is provided
            if args.output_file:
                import json
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output_file}")
                
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            
    else:
        # Interactive mode
        print("\nEnter feedback text to classify (type 'exit' to quit):")
        
        while True:
            text = input("\nFeedback text: ")
            
            if text.lower() in ['exit', 'quit', 'q']:
                break
                
            if not text.strip():
                continue
                
            prediction, probs, processed_text = predict_feedback(
                text, model, preprocessor, tokenizer, device
            )
            
            # Print results
            print(f"\nProcessed text: {processed_text}")
            print(f"Prediction: {'Valid' if prediction == 1 else 'Invalid'}")
            print(f"Confidence: {probs[prediction]:.4f}")
            print(f"Class probabilities:")
            print(f"  - Invalid: {probs[0]:.4f}")
            print(f"  - Valid: {probs[1]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with a trained My Zoom feedback validation model")
    
    parser.add_argument("--model_path", type=str, default="models/zoom_feedback_model.pt",
                        help="Path to the trained model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of the pretrained model used for training")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--input_file", type=str,
                        help="Path to file containing feedback texts (one per line)")
    parser.add_argument("--output_file", type=str,
                        help="Path to save prediction results (JSON format)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    
    main(args)
