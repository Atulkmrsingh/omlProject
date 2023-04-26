import os
import pandas as pd
import numpy as np
import csv
import PyPDF2
from submodlib import ProbabilisticSetCoverFunction
from PyPDF2 import PdfReader

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

def get_txt(pdf_path) :
    pdfReader = PdfReader(pdf_path)
    txt = "" 
    for page in range(len(pdfReader.pages)) : 
        pageObj = pdfReader.pages[page] 
        txt += pageObj.extract_text()
    return txt
def get_dict_words(path, src_lang, trans_lang) :
  dict_words = dict()
  files = os.listdir(path)
#   files = [file_ for file_ in files if file_.endswith(src_lang + "_" + trans_lang + ".csv")]
  for file_ in files :
    try :
      df = pd.read_csv(path + "/" + file_)
      words = df.loc[:,"English"]
    except : 
      continue
    dict_words[file_] = []
    for word in words :
      try :
        dict_words[file_].append(word.lower())
      except :
        continue
  
  return dict_words
def get_tokens(txt, tokenizer, lemmatizer, stop_words) :
    tokenized = tokenizer.tokenize(txt)
    statement_no_stop = [word.lower() for word in tokenized if word.lower() not in stop_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in statement_no_stop]
    return lemmatized

def get_idfs_samanantar(tokens) :
    csv_reader = csv.reader(open('./idf_samananthar.csv', 'r'))
    idf = dict()
    for row in csv_reader:
        k, v = row
        idf[k] = float(v)
    unknown_idf = max(list(idf.values()))
    weights = [idf[token] if token in idf else unknown_idf for token in tokens ]
    return weights
def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words) :
    n = len(tokens)
    coverage = [] #stores probability of every token ,tokens are concepts.
    z = set(tokens)
    for word in z : #for every word in pdf calculating the probability of being present in a dictionary
        prob = 0
        if word in dict_words:
          count =  tokens.count(word)
          prob = count/len(z)
        coverage.append(prob)     
    return np.array(coverage)


def select_glossaries2(pdf_path, src_lang, trans_lang, glossaries_path) :
    if (src_lang != "en" or trans_lang != "hi") :
        return None
    txt = get_txt(pdf_path)
    all_dict_words = get_dict_words(glossaries_path, src_lang, trans_lang)
    n= len(all_dict_words)
    budget = min(n,10)
    set_cover = get_set_cover(txt, all_dict_words, budget)
    dictionaries = list(all_dict_words.keys())
    selected_dictionaries = [dictionaries[_[0]][:-4] for _ in set_cover]
    return ",".join(selected_dictionaries)

def get_set_cover(source_text, all_dict_words, budget) :
    
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    tokens = get_tokens(source_text, tokenizer, lemmatizer, stop_words)
    z = set(tokens)
    weights = get_idfs_samanantar(z)
    
    
    
    # weights = get_idfs_samanantar(tokens)
    dict_coverage = []
    for name, dict_words in all_dict_words.items() :
            dict_coverage.append(get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words))
   

    n=len(all_dict_words)
    if budget >= n :
        budget = n - 1
    obj = ProbabilisticSetCoverFunction(n=n, probs=dict_coverage, num_concepts=len(z), concept_weights = weights)
    greedyList = obj.maximize(budget=budget, optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    
    return greedyList
    