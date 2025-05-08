
Designed and Developed by Gunalan


# My Zoom: A Transformer-Based Model for Contextual Feedback Validation

## Project Overview
This project implements a transformer-based model to validate contextual feedback for Zoom, an educational technology platform. The model classifies feedback as valid (1) or invalid (0) based on the content and context of the feedback.

## Skills Utilized
- Text Preprocessing and Data Augmentation
- Transformer Models (BERT)
- Binary Classification in NLP
- Model Evaluation and Performance Metrics
- Deployment using Gradio and Hugging Face Spaces

## Project Structure
- `data/`: Contains the training and evaluation datasets
- `notebooks/`: Jupyter notebooks for exploratory data analysis
- `src/`: Source code for the project
  - `data_preprocessing.py`: Functions for data preprocessing and augmentation
  - `model.py`: Transformer model implementation
  - `train.py`: Training script for the model
  - `evaluate.py`: Evaluation script
  - `utils.py`: Utility functions
- `app.py`: Gradio web application for model deployment
- `requirements.txt`: Required Python packages

## How to Use
1. Install dependencies: `pip install -r requirements.txt`
2. Preprocess data: `python src/data_preprocessing.py`
3. Train the model: `python src/train.py`
4. Evaluate the model: `python src/evaluate.py`
5. Run the web app: `python app.py`

## Model Details
The project uses a fine-tuned BERT model for binary classification of feedback text. The model processes the input text and determines whether the feedback is valid or not based on the context.

## Deployment
The model is deployed using Gradio, which provides an interactive web interface for users to input feedback and receive validation results.
