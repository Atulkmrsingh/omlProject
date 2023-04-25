import os
import pandas as pd
import numpy as np
import csv
import PyPDF2
from submodlib import FacilityLocationVariantMutualInformationFunction
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

def select_glossaries3(pdf_path, src_lang, trans_lang, glossaries_path) :
    if (src_lang != "en" or trans_lang != "hi") :
        return None
    txt = get_txt(pdf_path)
    all_dict_words = get_dict_words(glossaries_path, src_lang, trans_lang)
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    tokens = get_tokens(txt, tokenizer, lemmatizer, stop_words)
    idf = get_idfs_samanantar(tokens)
    n = len(all_dict_words)
    if n == 0:
        return None
    if n == 1:
        return list(all_dict_words.keys())[0][:-4]
    if n == 2:
        return ",".join([name[:-4] for name in all_dict_words.keys()])
    if n > 2:
        budget = min(n-1, 5)
        selected_glossaries = get_facility_location(all_dict_words, tokens, tokenizer, lemmatizer, stop_words, idf, budget)
        dictionaries = list(all_dict_words.keys())
        selected_dictionaries = [dictionaries[_[0]][:-4] for _ in selected_glossaries]
        return ",".join(selected_dictionaries)
def get_facility_location(all_dict_words, tokens, tokenizer, lemmatizer, stop_words, idf, budget):
    dict_coverage = []
    for name, dict_words in all_dict_words.items() :
        dict_coverage.append(get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words, idf))
    data = np.array(dict_coverage)
    queryData = np.ones((1, data.shape[1]))
    obj = FacilityLocationVariantMutualInformationFunction(n=data.shape[0], num_queries=1, data=data, queryData=queryData, metric="euclidean", queryDiversityEta=0.1)
    greedyList = obj.maximize(budget=budget,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    return greedyList

def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words, idf) :
    coverage = [] #stores probability of every token tokens are concepts.
    for word in tokens : #for every word in pdf calculating the probability of being present in a dictionary
        tokenized = [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(word) if word.lower() not in stop_words]
        if len(tokenized) == 0 :
            coverage.append(0)
            continue
        try :
            matched_inds = np.where(np.isin(dict_words, tokenized))
            count = np.sum(idf[matched_inds])
            prob = count/len(tokenized)
            coverage.append(prob)
        except :
            coverage.append(0)
            continue   
    return np.array(coverage)
