import pdfplumber
from submodlib import FacilityLocationVariantMutualInformationFunction
from submodlib import ProbabilisticSetCoverFunction
import time
import pandas as pd
import numpy as np
import csv
import PyPDF2

from PyPDF2 import PdfReader
import time
import os

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from itertools import combinations
def get_txt(pdf_path) :
    pdfReader = PdfReader(pdf_path)
    txt = "" 
    for page in range(len(pdfReader.pages)) : 
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

def select_glossaries3(pdf_path, src_lang, trans_lang, glossaries_path,budget) :
    if (src_lang != "en" or trans_lang != "hi") :
        return None
    txt = get_txt(pdf_path)
    all_dict_words,ind = get_dict_words(glossaries_path, src_lang, trans_lang)
    
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english')) 
    lemmatizer = WordNetLemmatizer()
    tokens = get_tokens(txt, tokenizer, lemmatizer, stop_words)
    z = set(tokens)
    idf = get_idfs_samanantar(z)
    n = len(all_dict_words)
    score =[]
    max_score =0 
    initial_set =[]
    dict_coverage = []
    for name, dict_words in all_dict_words.items() :
        dict_coverage.append(get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words, idf))
    k=0
    while(k<budget):
        
        max_score = 0
        copy = initial_set
        for entry in range(len(all_dict_words)):
            
            
            if entry not in initial_set:
                copy.append(entry)
                
                selected_glossaries_score1 = get_facility_location(all_dict_words, tokens, tokenizer, lemmatizer, stop_words, idf,copy,dict_coverage)
                selected_glossaries_score2 = probability_cover(all_dict_words,tokens,tokenizer,lemmatizer,stop_words,idf,copy,dict_coverage)
                sum = 0.8*selected_glossaries_score1 + 0.2*selected_glossaries_score2
            
                
                if sum > max_score:
                    element_to_be_added = entry
                    max_score = sum
        
                copy.remove(entry)
        initial_set.append(element_to_be_added)
        k=k+1
    for entry in initial_set:
     
     print(ind[entry])         
def get_facility_location(all_dict_words, tokens, tokenizer, lemmatizer, stop_words, idf,X,dict_coverage):
    
    data = np.array(dict_coverage)
   
    X = set(X)
    queryData = np.ones((1, data.shape[1]))
    obj = FacilityLocationVariantMutualInformationFunction(n=data.shape[0], num_queries=1, data=data, queryData=queryData, metric="euclidean", queryDiversityEta=0.1)
    x = obj.evaluate(X)
    return x
def probability_cover(all_dict_words, tokens, tokenizer, lemmatizer, stop_words, idf,X,dict_coverage) :
    
    z = set(tokens)
   
    n=len(all_dict_words)
    
   
    X= set(X)
    obj = ProbabilisticSetCoverFunction(n=n, probs=dict_coverage, num_concepts=len(z), concept_weights = idf)
    x = obj.evaluate(X)
    return x

def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words, idf) :
    n = len(tokens)
    coverage = [] #stores probability of every token ,tokens are concepts.
    z = set(tokens)
    for word in z : #for every word in pdf calculating the probability of being present in a dictionary
        prob = 0
        if word in dict_words:
          count =  tokens.count(word)
          prob = count/len(z)
        coverage.append(prob)     
    return tuple(coverage)
def isMachineReadable(pdf_file):
    try:
        pdf = pdfplumber.open(pdf_file)
    except:
        return

    for page_id in range(len(pdf.pages)):
        current_page = pdf.pages[page_id]
        words = current_page.extract_words()
        if(len(words)):
          break
    return len(words) > 0


# Returns preference order for english machine readable source, None otherwise
def get_preference_order(pdf_path, ocr_lang, src_lang, trans_lang, glossaries_path) :
    if src_lang != "en" or ocr_lang != "en" :
        return None

    if isMachineReadable(pdf_path) :
        #return select_glossaries1(pdf_path, src_lang, trans_lang, glossaries_path) #set cover
        return select_glossaries3(pdf_path, src_lang, trans_lang, glossaries_path,10)   #probability set cover
start = time.time()   
get_preference_order("animals.pdf","en","en","hi","en-hi_acronym_dicts_2")
end = time.time()
print(end-start)


    





