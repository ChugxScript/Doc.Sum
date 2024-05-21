# import os
# import nltk
# from nltk.corpus import stopwords
# from string import punctuation
# from docx import Document
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.tokenize import RegexpTokenizer
# from nltk.probability import FreqDist
# from heapq import nlargest

# def read_document(file_path):
#     doc = Document(file_path)
#     text = ""
#     original_sentences = []
#     for i, paragraph in enumerate(doc.paragraphs):
#         if len(paragraph.text.split(' ')) > 10:
#             print(f"\n>> paragraph {i+1}: {paragraph.text}")
#             text += paragraph.text + "\n"
#             original_sentences.extend(sent_tokenize(paragraph.text))
    
#     print(f"\n>> ORIGINAL TEXT: \n{text}")
#     return text, original_sentences

# def token(text):
#     tokenizer = RegexpTokenizer(r'\w+')
#     word_token = tokenizer.tokenize(text)
#     return word_token

# def sent_token(text):
#     sentence_token = sent_tokenize(text)
#     return sentence_token

# def remove_stopwords(word_list):
#     stopwords_list = set(stopwords.words('english'))
#     m_punctuation = punctuation + '\n'
#     filtered_words = [word for word in word_list if word.lower() not in stopwords_list and word.lower() not in stop_words_filipino and word not in m_punctuation]
#     return filtered_words

# def word_freq(filtered_words):
#     fdist = FreqDist(filtered_words)
#     return fdist

# def max_freq(word_frequency):
#     max_freq = max(word_frequency.values())
#     for word in word_frequency:
#         word_frequency[word] /= max_freq
#     return word_frequency

# def sentence_scores(sent_list, word_frequency):
#     sent_scores = {}
#     for sentence in sent_list:
#         for word in word_tokenize(sentence.lower()):
#             if word in word_frequency:
#                 if sentence not in sent_scores:
#                     sent_scores[sentence] = word_frequency[word]
#                 else:
#                     sent_scores[sentence] += word_frequency[word]
#     return sent_scores

# def get_summary(sent_scores, original_sentences):
#     summary = nlargest(10, sent_scores, key=sent_scores.get)
#     summary.sort(key=lambda sentence: original_sentences.index(sentence))
#     final_summary = ' '.join(summary)
#     return final_summary

# DOCENGDOCX = os.path.join("Others", "DOC_TAGALOG.docx")
# stop_words_filipino = {
#     'at', 'ang', 'Ang', 'sa', 'ng', 'mga', 'ngunit', 'o', 'pero', 'kaya', 'din',
#     'rin', 'ito', 'yan', 'ni', 'si', 'sina', 'nila', 'kay', 'mula', 'hanggang',
#     'kung', 'dahil', 'sapagkat', 'dahilan', 'kaya', 'tulad', 'tulad ng',
#     'katulad', 'kung paano', 'kung paanong', 'gayon', 'gaya', 'gaya ng',
#     'nang', 'tuwing', 'kapag', 'noon', 'bago', 'pagkatapos', 'habang', 'samantalang',
#     'kayat', 'upang', 'para', 'nang', 'nang sa gayon', 'upang', 'dahil sa',
#     'kaya', 'upang', 'nang', 'pagkatapos', 'upang', 'kahit', 'kahit na', 'maging',
#     'kaysa', 'sa halip', 'kumpara sa', 'bagama\'t', 'bagaman', 'bagama\'t',
#     'bagama\'t', 'saka', 'sakali', 'nga', 'dahil', 'nga kung', 'sila', 'siya',
#     'ang kaniyang', 'kanyang', 'kaniya', 'nila', 'nilang', 'sa kanila', 'sa kanilang',
#     'kanila', 'kanilang', 'ngunit', 'subalit', 'dati', 'noon', 'noon pa', 'ngayon',
#     'ngayon pa', 'simula', 'simula noong', 'hanggang', 'hanggang sa', 'hindi',
#     'hindi kailanman', 'kahit kailan', 'kailanman', 'walang', 'wala', 'wala nang',
#     'kailanman', 'wala', 'wala nang', 'dito', 'doon', 'doon', 'doon', 'rito',
#     'roon', 'kung saan', 'saan', 'kung saan-saan', 'ano', 'sino', 'paano', 'kanino',
#     'alin', 'saan', 'kailan', 'paano', 'magkano', 'gaano', 'gaanong', 'ilang', 'ilang-ilang',
#     'ano-anong', 'sino-sino', 'anong', 'ilang-ilang', 'maraming', 'marami', 'ilan', 'ibang',
#     'iba', 'iba\'t ibang', 'isa', 'isang', 'isa\'t isa', 'isang', 'isa\'t', 'isa', 'iba',
#     'ibang', 'nasaan', 'nasa', 'narito', 'doon', 'riyan', 'dito', 'rito', 'nandito',
#     'nandiyan', 'nandyan', 'narito', 'niyan', 'niyan', 'niyan', 'niyan', 'niyan',
#     'niyan', 'niyan', 'niyan', 'niyan', 'iyon', 'iyon', 'iyon', 'iyon', 'iyon', 'iyon',
#     'iyon', 'iyon', 'kanya', 'kaniya', 'sa kaniya', 'kanya', 'niya', 'niya', 'niya',
#     'kanya', 'kanila', 'sa kanila', 'kanilang', 'kami', 'tayo', 'kayo', 'sila', 'sarili',
#     'amin', 'na', 'ay', 'may', 'ating', 'pag', 'di',
# }

# docu, original_sentences = read_document(DOCENGDOCX)
# tokenized_word_docu = token(docu)
# tokenized_sentence_docu = sent_token(docu)
# filtered_word_docu = remove_stopwords(tokenized_word_docu)
# word_docu_freq = word_freq(filtered_word_docu)
# sorted_words = sorted(word_docu_freq.items(), key=lambda x: x[1], reverse=True)
# print(f"\n>> [1]sorted_words: \n{sorted_words}")

# word_docu_freq = max_freq(word_docu_freq)
# sentence_score_docu = sentence_scores(tokenized_sentence_docu, word_docu_freq)
# sorted_sentences = sorted(sentence_score_docu.items(), key=lambda x: x[1], reverse=True)
# print("\n>> sentence_score_docu (sorted by score):")
# for sentence, score in sorted_sentences:
#     print(f"{sentence}: {score}")

# nltk_summary = get_summary(sentence_score_docu, original_sentences)
# print(f"\n>> nltk_summary: \n{nltk_summary}")

# from transformers import pipeline
# from PyPDF2 import PdfReader
# import os

# # Read PDF File
# def read_pdf(file_path):
#     with open(file_path, 'rb') as pdf_file:
#         read_pdf = PdfReader(pdf_file)
#         number_of_pages = len(read_pdf.pages)
#         text = ''
#         for page_number in range(number_of_pages):
#             page = read_pdf.pages[page_number]
#             page_content = page.extract_text()
#             if page_content:  # Ensure there's text on the page
#                 text += page_content
#     return text

# # Chunk text into smaller parts if needed
# def chunk_text(text, max_length=1000):
#     words = text.split()
#     chunks = []
#     current_chunk = []
#     current_length = 0

#     for word in words:
#         current_length += len(word) + 1  # +1 for the space
#         if current_length > max_length:
#             chunks.append(' '.join(current_chunk))
#             current_chunk = [word]
#             current_length = len(word) + 1
#         else:
#             current_chunk.append(word)
    
#     if current_chunk:  # Add the last chunk
#         chunks.append(' '.join(current_chunk))
    
#     return chunks

# # Main Execution
# # file_path = os.path.join("others", "DOC_ENGLISH_ESSAY.pdf")
# file_path = os.path.join("others", "DOC_TAGALOG.pdf")
# document_text = read_pdf(file_path)

# # # Initialize the summarization pipeline
# # pipe = pipeline('summarization', model='facebook/bart-large')

# # # Summarize each chunk and combine results
# # document_chunks = chunk_text(document_text)
# # summaries = [pipe(chunk)[0]['summary_text'] for chunk in document_chunks]

# # # Combine summaries into a single summary
# # final_summary = ' '.join(summaries)
# # print(f"Summary: \n{final_summary}")


# def chunk_text(text, tokenizer, chunk_size):
#     inputs = tokenizer.encode(text, return_tensors='pt', max_length=chunk_size, truncation=True, padding=True)
#     input_ids = inputs[0]
#     chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
#     return chunks

# def summarize_chunk(chunk, model, tokenizer, max_length):
#     summary_ids = model.generate(chunk.unsqueeze(0), max_length=max_length, min_length=max_length // 2, num_beams=4, length_penalty=2.0, early_stopping=True)
#     summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#     return summary

# from concurrent.futures import ThreadPoolExecutor
# def BART_summary(document, model, tokenizer, chunk_size, summary_percentage):
#     chunks = chunk_text(document, tokenizer, chunk_size)
#     total_tokens = sum([len(chunk) for chunk in chunks])
#     max_length = int(chunk_size * (summary_percentage / 100))
    
#     with ThreadPoolExecutor() as executor:
#         summaries = list(executor.map(lambda chunk: summarize_chunk(chunk, model, tokenizer, max_length), chunks))
    
#     return " ".join(summaries)

# from transformers import BartForConditionalGeneration, BartTokenizer
# model_name = 'facebook/bart-large-cnn'
# model = BartForConditionalGeneration.from_pretrained(model_name)
# tokenizer = BartTokenizer.from_pretrained(model_name)
# BART_sum_perc = 30
# BART_chunk_sz = 1024
# bart_summary = BART_summary(document_text, model, tokenizer, BART_chunk_sz, BART_sum_perc)
# print(f"\n>>bart_summary: \n{bart_summary}")

from datasets import load_dataset
import os

train_path = os.path.join("Others", "train_data.json")
val_path = os.path.join("Others", "val_data.json")
test_path = os.path.join("Others", "test_data.json")

# Load your JSON dataset
dataset = load_dataset('json', data_files={'train': train_path, 'validation': val_path, 'test': test_path})

# Optionally, you can inspect the dataset
print(dataset)

from datasets import DatasetDict, Dataset
from huggingface_hub import HfApi

# Example of creating a DatasetDict (if your dataset is not already in this format)
dataset_dict = DatasetDict({
    'train': dataset['train'],
    'validation': dataset['validation'],
    'test': dataset['test']
})

# Push the dataset to the Hugging Face Hub
dataset_dict.push_to_hub("test_summary")

# Or if you are uploading a single dataset
# dataset.push_to_hub("test_summary")
