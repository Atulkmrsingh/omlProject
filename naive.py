import os
import pandas as pd
import numpy as np
import csv
import PyPDF2
import pdfplumber
from submodlib import SetCoverFunction
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
import time

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def get_txt(pdf_path):
    pdfReader = PdfReader(pdf_path)
    txt = "" 
    for page in range(len(pdfReader.pages)):
        pageObj = pdfReader.pages[page]
        txt += pageObj.extract_text()
    return txt

def get_dict_words(path, src_lang, trans_lang):
    dict_words = dict()
    files = os.listdir(path)
    ind = {}
    i = 0
    for file_ in files:
        try:
            df = pd.read_csv(path + "/" + file_)
            words = df.loc[:,"English"]
        except:
            continue
        dict_words[file_] = []
        ind[i] = file_
        i = i + 1
        for word in words:
            try:
                dict_words[file_].append(word.lower())
            except:
                continue
    return dict_words, ind

def get_tokens(txt, tokenizer):
    tokenized = tokenizer.tokenize(txt)
    statement = [word.lower() for word in tokenized]
    return statement


def get_vectors_for_dictionary(source_text, all_dict_words):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = get_tokens(source_text, tokenizer)
    similarity_matrix = []
    for dict in all_dict_words:
        dictionary = all_dict_words[dict]
        binary_vector = [1 if word in dictionary else 0 for word in tokens]
        similarity_matrix.append(binary_vector)
    binary_vector = np.array(similarity_matrix)
    return similarity_matrix


all_dict_words, ind = get_dict_words("en-hi_acronym_dicts_2", "en", "hi")
source_text = get_txt("pdf24_merged.pdf")

start_time = time.time()
g = get_vectors_for_dictionary(source_text, all_dict_words)

budget = 10
ind_budget = [sum(lst) for lst in g]
sorted_sums = sorted(ind_budget, reverse=True)[:budget]
top_k_indices = [ind_budget.index(sum_val) for sum_val in sorted_sums]
for entry in top_k_indices:
    print(ind[entry])

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds.")
