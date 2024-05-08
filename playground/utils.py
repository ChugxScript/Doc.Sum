# playground/utils.py
from docx import Document
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize

def extract_text_from_pdf(uploaded_file):
    text = ""
    with uploaded_file.open('rb') as f:
        pdf_reader = PdfReader(f)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            sentences = sent_tokenize(extracted_text)
            for sentence in sentences:
                if len(sentence.split(' ')) > 10:
                    text += sentence + "\n"
    return text


def extract_text_from_docx(uploaded_file):
    text = ""
    doc = Document(uploaded_file)
    for i, paragraph in enumerate(doc.paragraphs):
        if len(paragraph.text.split(' ')) > 10:
            text += paragraph.text + "\n"
    return text

def extract_text_from_txt(uploaded_file):
    text = uploaded_file.read().decode('utf-8')
    original_sentences = [sentence for sentence in sent_tokenize(text) if len(sentence.split(' ')) > 10]
    
    # update text
    text = ''
    for sentence in original_sentences:
        text += sentence + '\n'
    return text

def format_input_text(input_text):
    original_sentences = [sentence for sentence in sent_tokenize(input_text) if len(sentence.split(' ')) > 10]
    
    # update text
    text = ''
    for sentence in original_sentences:
        text += sentence + '\n'
    return text