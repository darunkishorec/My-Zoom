"""
Setup script for the My Zoom project.
Creates necessary directories and prepares the environment.
"""

import os
import sys

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
            return True
        except Exception as e:
            print(f"Error creating directory {directory_path}: {str(e)}")
            return False
    else:
        print(f"Directory already exists: {directory_path}")
        return True

def setup_project():
    """Set up the project directory structure."""
    # Get the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories
    create_directory(os.path.join(project_dir, "models"))
    create_directory(os.path.join(project_dir, "evaluation"))
    
    # Check if required files exist
    required_files = ["train.xlsx", "evaluation.xlsx"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(project_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print("\nWarning: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure these files are in the project directory before running the model.")
    else:
        print("\nAll required data files are present.")
    
    # Create a virtual environment instructions file
    venv_instructions = """
# Setting up a Virtual Environment for the My Zoom Project

## Windows
```
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\\Scripts\\activate

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
"""
    venv_path = os.path.join(project_dir, "venv_setup.md")
    with open(venv_path, "w") as f:
        f.write(venv_instructions)
    print(f"Created virtual environment setup instructions: {venv_path}")
    
    print("\nProject setup complete!")

if __name__ == "__main__":
    setup_project()
