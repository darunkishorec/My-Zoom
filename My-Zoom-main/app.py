"""
Gradio web application for the My Zoom feedback validation model.
"""

import os
import torch
import gradio as gr
from transformers import BertTokenizer
from src.model import ZoomFeedbackClassifier
from src.data_preprocessing import DataPreprocessor
from src.utils import preprocess_input_text

# Define paths
MODEL_DIR = "models"
DEFAULT_MODEL_PATH = None

# Find the latest model file
if os.path.exists(MODEL_DIR):
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
    if model_files:
        model_files.sort(reverse=True)  # Sort by name (assumes timestamp in filename)
        DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, model_files[0])

# If no model is found, use a default path that will be checked for existence
if DEFAULT_MODEL_PATH is None:
    DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "zoom_feedback_model.pt")

# Define model parameters
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Initialize preprocessor
preprocessor = DataPreprocessor(max_length=MAX_LENGTH, tokenizer_name=MODEL_NAME)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model if exists
def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the trained model if it exists."""
    if os.path.exists(model_path):
        model = ZoomFeedbackClassifier(pretrained_model_name=MODEL_NAME)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    return None

model = load_model()

def classify_feedback(text, model_path=DEFAULT_MODEL_PATH):
    """
    Classify the input text as valid or invalid feedback.
    
    Args:
        text (str): Input feedback text
        model_path (str): Path to the model file
        
    Returns:
        tuple: (Predicted class, Class probabilities, Processed text)
    """
    # Load model if not already loaded or if a different model_path is provided
    global model
    if model is None or model_path != DEFAULT_MODEL_PATH:
        model = load_model(model_path)
        if model is None:
            return "No model found", {"Valid": 0.0, "Invalid": 0.0}, text
    
    # Preprocess the text
    cleaned_text = preprocessor.clean_text(text)
    
    # Tokenize the text
    inputs = tokenizer(
        cleaned_text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    # Create results
    class_probs = {
        "Invalid": round(float(probabilities[0, 0]), 4),
        "Valid": round(float(probabilities[0, 1]), 4)
    }
    
    predicted_class = "Valid" if prediction == 1 else "Invalid"
    
    return predicted_class, class_probs, cleaned_text

def get_feedback_examples():
    """Return some example feedback texts."""
    return [
        "This app is very useful for online classes and meetings",
        "I cannot download this app, it keeps crashing",
        "Very practical and easy to use for remote work",
        "Zoom is the best app for video conferencing",
        "The app crashed when I tried to join a meeting"
    ]

# Create Gradio interface
with gr.Blocks(title="My Zoom: Feedback Validation") as demo:
    gr.Markdown("# My Zoom: Feedback Validation")
    gr.Markdown("""
    This application validates feedback about the Zoom application, classifying it as either Valid or Invalid.
    Enter your feedback text below and click 'Submit' to get the validation result.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Feedback Text",
                placeholder="Enter your feedback about Zoom here...",
                lines=5
            )
            
            examples = gr.Examples(
                examples=get_feedback_examples(),
                inputs=input_text
            )
            
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            output_class = gr.Textbox(label="Validation Result")
            output_probs = gr.JSON(label="Class Probabilities")
            output_processed = gr.Textbox(label="Processed Text")
    
    submit_button.click(
        classify_feedback,
        inputs=[input_text],
        outputs=[output_class, output_probs, output_processed]
    )
    
    gr.Markdown("""
    ## How It Works
    
    This application uses a fine-tuned BERT model to classify feedback text. The model has been trained on a dataset of Zoom app feedback.
    
    1. **Input**: Your feedback text about the Zoom application
    2. **Processing**: The text is cleaned and tokenized
    3. **Classification**: The model predicts whether the feedback is Valid or Invalid
    4. **Output**: The prediction result, class probabilities, and the processed text
    
    ## About
    
    This project implements a transformer-based model for contextual feedback validation in the EdTech domain.
    """)

# Launch the app
if __name__ == "__main__":
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created directory: {MODEL_DIR}")
    
    # Check if model exists
    if model is None:
        print(f"Warning: No model found at {DEFAULT_MODEL_PATH}")
        print("You can train a model using the training script before running the app.")
        print("The app will still run but won't provide accurate classifications.")
    
    # Launch Gradio app
    demo.launch(share=True)
