# playground/utils.py
from docx import Document
import os

def extract_text_from_document(file):
    _, file_extension = os.path.splitext(file.name)
    if file_extension.lower() == '.docx':
        document = Document(file)
        text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
    elif file_extension.lower() == '.txt':
        text = file.read().decode('utf-8')
    else:
        text = 'Unsupported file format'
    return text

def extract_text_from_pdf(uploaded_file):
    # Import PdfReader from PyPDF2
    from PyPDF2 import PdfReader
    
    # Initialize an empty string to store the extracted text
    text = ""

    # Open the PDF file in binary mode and create a PdfReader object
    with uploaded_file.open('rb') as f:
        pdf_reader = PdfReader(f)

        # Iterate through each page and extract text
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def extract_text_from_docx(uploaded_file):
    # Implement DOCX text extraction logic here
    # Example using python-docx library
    import docx
    text = ""
    doc = docx.Document(uploaded_file)
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text