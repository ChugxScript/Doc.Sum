import os
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from heapq import nlargest
from transformers import BartForConditionalGeneration, BartTokenizer
from concurrent.futures import ThreadPoolExecutor

# nltk.download('stopwords')
# nltk.download('punkt')
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def summarize(uploaded_text, summary_len):
    tokenized_text = token(uploaded_text)
    tokenized_sentence, orig_sent = sent_token(uploaded_text)
    filter_text = remove_stopwords(tokenized_text)
    text_freq = word_freq(filter_text)
    text_freq = max_freq(text_freq)
    rank_sentence = sentence_scores(tokenized_sentence, text_freq)
    nltk_summary = get_summary(rank_sentence, orig_sent, summary_len)

    BART_sum_perc = summary_len
    BART_chunk_sz = 1024
    bart_summary = BART_summary(uploaded_text, model, tokenizer, BART_chunk_sz, BART_sum_perc)
    
    index = check_index(bart_summary)
    reference_summary = get_sample_from_dataset(dataset, index)
    scores = evaluate_summary(reference_summary, nltk_summary)
    print(f"nltk_summary score: {scores}")

    scores = evaluate_summary(reference_summary, bart_summary)
    print(f"bart_summary score: {scores}")
    return nltk_summary, bart_summary

def token(text):
    tokenizer = RegexpTokenizer(r'\w+')
    word_token = tokenizer.tokenize(text)
    return word_token

def sent_token(text):
    sentence_token = sent_tokenize(text)

    original_sentences = []
    for sent in sentence_token:
        original_sentences.append(sent)
    return sentence_token, original_sentences

def remove_stopwords(word_list):
    stopwords_list = set(stopwords.words('english'))
    m_punctuation = punctuation + '\n'
    filtered_words = [word for word in word_list if word.lower() not in stopwords_list and word.lower() not in stop_words_filipino and word not in m_punctuation]
    return filtered_words

def word_freq(filtered_words):
    fdist = FreqDist(filtered_words)
    return fdist

def max_freq(word_frequency):
    max_freq = max(word_frequency.values())
    for word in word_frequency:
        word_frequency[word] /= max_freq
    return word_frequency

def sentence_scores(sent_list, word_frequency):
    sent_scores = {}
    for sentence in sent_list:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequency:
                if sentence not in sent_scores:
                    sent_scores[sentence] = word_frequency[word]
                else:
                    sent_scores[sentence] += word_frequency[word]
    return sent_scores

def get_summary(sent_scores, original_sentences, summary_len):
    summary = nlargest(summary_len, sent_scores, key=sent_scores.get)
    summary.sort(key=lambda sentence: original_sentences.index(sentence))
    final_summary = ' '.join(summary)
    return final_summary

def chunk_text(text, tokenizer, chunk_size):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=chunk_size, truncation=True, padding=True)
    input_ids = inputs[0]
    chunks = [input_ids[i:i + chunk_size] for i in range(0, len(input_ids), chunk_size)]
    return chunks

def summarize_chunk(chunk, model, tokenizer, max_length):
    summary_ids = model.generate(chunk.unsqueeze(0), max_length=max_length, min_length=max_length // 2, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def BART_summary(document, model, tokenizer, chunk_size, summary_percentage):
    chunks = chunk_text(document, tokenizer, chunk_size)
    total_tokens = sum([len(chunk) for chunk in chunks])
    max_length = int(chunk_size * (summary_percentage / 100))
    
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(lambda chunk: summarize_chunk(chunk, model, tokenizer, max_length), chunks))
    
    return " ".join(summaries)

stop_words_filipino = {
    'at', 'ang', 'Ang', 'sa', 'ng', 'mga', 'ngunit', 'o', 'pero', 'kaya', 'din',
    'rin', 'ito', 'yan', 'ni', 'si', 'sina', 'nila', 'kay', 'mula', 'hanggang',
    'kung', 'dahil', 'sapagkat', 'dahilan', 'kaya', 'tulad', 'tulad ng',
    'katulad', 'kung paano', 'kung paanong', 'gayon', 'gaya', 'gaya ng',
    'nang', 'tuwing', 'kapag', 'noon', 'bago', 'pagkatapos', 'habang', 'samantalang',
    'kayat', 'upang', 'para', 'nang', 'nang sa gayon', 'upang', 'dahil sa',
    'kaya', 'upang', 'nang', 'pagkatapos', 'upang', 'kahit', 'kahit na', 'maging',
    'kaysa', 'sa halip', 'kumpara sa', 'bagama\'t', 'bagaman', 'bagama\'t',
    'bagama\'t', 'saka', 'sakali', 'nga', 'dahil', 'nga kung', 'sila', 'siya',
    'ang kaniyang', 'kanyang', 'kaniya', 'nila', 'nilang', 'sa kanila', 'sa kanilang',
    'kanila', 'kanilang', 'ngunit', 'subalit', 'dati', 'noon', 'noon pa', 'ngayon',
    'ngayon pa', 'simula', 'simula noong', 'hanggang', 'hanggang sa', 'hindi',
    'hindi kailanman', 'kahit kailan', 'kailanman', 'walang', 'wala', 'wala nang',
    'kailanman', 'wala', 'wala nang', 'dito', 'doon', 'doon', 'doon', 'rito',
    'roon', 'kung saan', 'saan', 'kung saan-saan', 'ano', 'sino', 'paano', 'kanino',
    'alin', 'saan', 'kailan', 'paano', 'magkano', 'gaano', 'gaanong', 'ilang', 'ilang-ilang',
    'ano-anong', 'sino-sino', 'anong', 'ilang-ilang', 'maraming', 'marami', 'ilan', 'ibang',
    'iba', 'iba\'t ibang', 'isa', 'isang', 'isa\'t isa', 'isang', 'isa\'t', 'isa', 'iba',
    'ibang', 'nasaan', 'nasa', 'narito', 'doon', 'riyan', 'dito', 'rito', 'nandito',
    'nandiyan', 'nandyan', 'narito', 'niyan', 'niyan', 'niyan', 'niyan', 'niyan',
    'niyan', 'niyan', 'niyan', 'niyan', 'iyon', 'iyon', 'iyon', 'iyon', 'iyon', 'iyon',
    'iyon', 'iyon', 'kanya', 'kaniya', 'sa kaniya', 'kanya', 'niya', 'niya', 'niya',
    'kanya', 'kanila', 'sa kanila', 'kanilang', 'kami', 'tayo', 'kayo', 'sila', 'sarili',
    'amin', 'na', 'ay', 'may', 'ating', 'pag', 'di',
}

import datasets
dataset = datasets.load_dataset('olors/test_summary')

def get_sample_from_dataset(dataset, index):
    sample = dataset['train'][index]
    reference_summary = sample['summary']
    return reference_summary

from langdetect import detect

def check_index(text):
    try:
        language = detect(text)
        if language == 'en':
            return 0
        else:
            return 1
    except:
        return 0

from rouge_score import rouge_scorer

def evaluate_summary(reference_summary, generated_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    return scores

