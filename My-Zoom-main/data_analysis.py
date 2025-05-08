import pandas as pd
import os

# Define the directory and file paths
project_dir = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(project_dir, "train.xlsx")
eval_file = os.path.join(project_dir, "evaluation.xlsx")
pdf_file = os.path.join(project_dir, "My Zoom.pdf")

def analyze_excel_file(file_path):
    """Analyze an Excel file and return information about its structure"""
    print(f"\nAnalyzing file: {os.path.basename(file_path)}")
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        
        # Print basic information
        print(f"Number of rows: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Print data types for each column
        print("\nColumn data types:")
        for col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        
        # Print a sample of the data
        print("\nFirst 5 rows:")
        print(df.head(5))
        
        return df
    except Exception as e:
        print(f"Error analyzing file {file_path}: {str(e)}")
        return None

def analyze_pdf_file(file_path):
    """Attempt to extract text from PDF file"""
    try:
        import PyPDF2
        print(f"\nAttempting to read PDF file: {os.path.basename(file_path)}")
        
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            print(f"Number of pages: {num_pages}")
            
            # Extract text from first page
            if num_pages > 0:
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                print("\nFirst page content preview (first 500 chars):")
                print(text[:500] + "..." if len(text) > 500 else text)
    except Exception as e:
        print(f"Error reading PDF file {file_path}: {str(e)}")

if __name__ == "__main__":
    print("Data Analysis for Project")
    print("=" * 50)
    
    # Analyze train.xlsx
    train_df = analyze_excel_file(train_file)
    
    # Analyze evaluation.xlsx
    eval_df = analyze_excel_file(eval_file)
    
    # Try to extract information from PDF
    analyze_pdf_file(pdf_file)
