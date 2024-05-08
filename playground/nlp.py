import os
import nltk
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from heapq import nlargest

# nltk.download('stopwords')
# nltk.download('punkt')

def summarize(uploaded_text, summary_len):
    tokenized_text = token(uploaded_text)
    tokenized_sentence, orig_sent = sent_token(uploaded_text)
    filter_text = remove_stopwords(tokenized_text)
    text_freq = word_freq(filter_text)
    text_freq = max_freq(text_freq)
    rank_sentence = sentence_scores(tokenized_sentence, text_freq)
    nltk_summary = get_summary(rank_sentence, orig_sent, summary_len)

    return nltk_summary

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





# import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from string import punctuation

# # pip install python-docx
# from docx import Document
# from heapq import nlargest

# # python -m spacy download en_core_web_sm

# def summarize(file_content, summary_len):
#     # create a list of stopwords.
#     stopwords = list(STOP_WORDS)
#     tokenize_doc = spacy.load('en_core_web_sm')

#     # tokenizes the text, i.e. segments it into words, punctuation and so on
#     file = tokenize_doc(file_content)
#     tokens = [token.text for token in file]
#     modified_punctuation = punctuation + '\n'

#     word_freqs = word_frequency_table(file, stopwords, modified_punctuation)
#     max_frequency = max(word_freqs.values())
#     normalize_word_freqs = normalize_frequency(word_freqs, max_frequency)
#     tokenized_sentence = [sent for sent in file.sents]
#     sentence_scores = sentence_score(tokenized_sentence, normalize_word_freqs)
#     summary_length = int(len(tokenized_sentence)*summary_len)
#     init_summary = nlargest(summary_length, sentence_scores, key = sentence_scores.get)

#     filter_summary = [word.text for word in init_summary]
#     final_summary = ' '.join(filter_summary)
#     # print("\nDoc:\n", file)
#     # print("\n\nfinal_summary: \n", final_summary)
    
#     # print("\nDoc len:", len(file))
#     # print("\nfinal_summary len:", len(final_summary))

#     return final_summary

# def word_frequency_table(docz, stopwordsz, punctuationz):
#     word_frequencies = {}
#     for word in docz:
#         if word.text.lower() not in stopwordsz:
#             if word.text.lower() not in punctuationz:
#                 if word.text not in word_frequencies.keys():
#                     word_frequencies[word.text] = 1
#                 else:
#                     word_frequencies[word.text] += 1
#     return word_frequencies

# def normalize_frequency(word_freqsz, max_frequencyz):
#     for word in word_freqsz.keys():
#         word_freqsz[word] = word_freqsz[word]/max_frequencyz
#     return word_freqsz

# def sentence_score(tokenized_sentencez, normalize_word_freqsz):
#     sentence_scores = {}
#     for sent in tokenized_sentencez:
#         for word in sent:
#             if word.text.lower() in normalize_word_freqsz.keys():
#                 if sent not in sentence_scores.keys():
#                     sentence_scores[sent] = normalize_word_freqsz[word.text.lower()]
#                 else:
#                     sentence_scores[sent] += normalize_word_freqsz[word.text.lower()]
#     return sentence_scores
