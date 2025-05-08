import PyPDF2
import os

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file and return it as a string"""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n\n--- Page Break ---\n\n"
        return text

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "My Zoom.pdf")
    
    try:
        extracted_text = extract_pdf_text(pdf_path)
        print("PDF Content:")
        print("=" * 50)
        print(extracted_text)
        
        # Also save the text to a file for reference
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdf_content.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Text also saved to {output_path}")
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
