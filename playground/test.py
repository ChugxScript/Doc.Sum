import os
import nltk
from nltk.corpus import stopwords
from string import punctuation
from docx import Document
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from heapq import nlargest

def read_document(file_path):
    doc = Document(file_path)
    text = ""
    original_sentences = []
    for i, paragraph in enumerate(doc.paragraphs):
        if len(paragraph.text.split(' ')) > 10:
            print(f"\n>> paragraph {i+1}: {paragraph.text}")
            text += paragraph.text + "\n"
            original_sentences.extend(sent_tokenize(paragraph.text))
    
    print(f"\n>> ORIGINAL TEXT: \n{text}")
    return text, original_sentences

def token(text):
    tokenizer = RegexpTokenizer(r'\w+')
    word_token = tokenizer.tokenize(text)
    return word_token

def sent_token(text):
    sentence_token = sent_tokenize(text)
    return sentence_token

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

def get_summary(sent_scores, original_sentences):
    summary = nlargest(10, sent_scores, key=sent_scores.get)
    summary.sort(key=lambda sentence: original_sentences.index(sentence))
    final_summary = ' '.join(summary)
    return final_summary

DOCENGDOCX = os.path.join("Others", "DOC_TAGALOG.docx")
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

docu, original_sentences = read_document(DOCENGDOCX)
tokenized_word_docu = token(docu)
tokenized_sentence_docu = sent_token(docu)
filtered_word_docu = remove_stopwords(tokenized_word_docu)
word_docu_freq = word_freq(filtered_word_docu)
sorted_words = sorted(word_docu_freq.items(), key=lambda x: x[1], reverse=True)
print(f"\n>> [1]sorted_words: \n{sorted_words}")

word_docu_freq = max_freq(word_docu_freq)
sentence_score_docu = sentence_scores(tokenized_sentence_docu, word_docu_freq)
sorted_sentences = sorted(sentence_score_docu.items(), key=lambda x: x[1], reverse=True)
print("\n>> sentence_score_docu (sorted by score):")
for sentence, score in sorted_sentences:
    print(f"{sentence}: {score}")

nltk_summary = get_summary(sentence_score_docu, original_sentences)
print(f"\n>> nltk_summary: \n{nltk_summary}")
