
Designed and Developed by Gunalan

# My Zoom: Quick Start Guide

This guide provides step-by-step instructions to get started with the My Zoom feedback validation project.

## 1. Setup

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, run the setup script to prepare the project environment:

```bash
python setup.py
```

## 2. Data Exploration

To understand the dataset structure and characteristics:

```bash
python exploratory_data_analysis.py
```

This will generate visualizations showing class distribution, text length patterns, and word frequencies in the feedback data.

## 3. Training the Model

Train the BERT-based model on the provided data:

```bash
python src/train.py
```

Optional arguments:
- `--epochs 6` (default: 4) - Number of training epochs
- `--batch_size 32` (default: 16) - Batch size for training
- `--augment_data` - Augment minority class data to balance classes
- `--learning_rate 3e-5` (default: 2e-5) - Learning rate for training

Example with custom parameters:
```bash
python src/train.py --epochs 6 --batch_size 32 --augment_data --learning_rate 3e-5
```

## 4. Evaluating the Model

After training, evaluate the model's performance:

```bash
python src/evaluate.py --model_path models/zoom_feedback_model_{timestamp}.pt
```

Replace `{timestamp}` with the timestamp in your saved model file.

Optional arguments:
- `--analyze_errors` - Analyze misclassified examples
- `--output_dir evaluation` - Directory to save evaluation results

## 5. Making Predictions

### Interactive Mode
For interactive predictions on individual feedback texts:

```bash
python predict.py
```

### Batch Processing
For processing multiple feedback texts from a file:

```bash
python batch_process.py --input_file your_data.xlsx --output_file results.xlsx
```

## 6. Fine-tuning with New Data

To adapt the model to new data:

```bash
python finetune.py --new_data new_feedback.xlsx --base_model_path models/zoom_feedback_model.pt
```

## 7. Web Application

Launch the Gradio web application for an interactive interface:

```bash
python app.py
```

This will start a local web server and provide a URL to access the application.

## Example: Complete Workflow

Here's a full example workflow to get from setup to deployment:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore data
python exploratory_data_analysis.py

# 3. Train model
python src/train.py --augment_data --epochs 5

# 4. Evaluate model (assuming model saved as models/zoom_feedback_model_20250508_170000.pt)
python src/evaluate.py --model_path models/zoom_feedback_model_20250508_170000.pt --analyze_errors

# 5. Run the web app
python app.py
```

## Project Structure

- `src/`: Core source code
  - `data_preprocessing.py`: Data processing pipeline
  - `model.py`: BERT model implementation
  - `train.py`: Training script
  - `evaluate.py`: Evaluation script
  - `utils.py`: Utility functions
- `app.py`: Gradio web application
- `predict.py`: Interactive prediction script
- `batch_process.py`: Batch processing script
- `finetune.py`: Model fine-tuning script
- `exploratory_data_analysis.py`: Data analysis script
- `models/`: Saved model files
- `evaluation/`: Evaluation results
