
Designed and Developed by Gunalan

# Setting up a Virtual Environment for the My Zoom Project

## Windows
```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## macOS/Linux
```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running the Project
After setting up the environment:

1. Run exploratory data analysis:
   python exploratory_data_analysis.py

2. Train the model:
   python src/train.py

3. Evaluate the model:
   python src/evaluate.py --model_path models/zoom_feedback_model.pt

4. Run the web app:
   python app.py
