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
  ind={} # stores the index corresponding to the dictionary
  i = 0
#   files = [file_ for file_ in files if file_.endswith(src_lang + "_" + trans_lang + ".csv")]
  for file_ in files :
    try :
      df = pd.read_csv(path + "/" + file_)
      words = df.loc[:,"English"]
    except : 
      continue
    dict_words[file_] = []
    ind[i]= file_
    i = i+1
    for word in words :
      try :
        dict_words[file_].append(word.lower())
      except :
        continue
  
  return dict_words,ind

def get_tokens(txt, tokenizer, lemmatizer, stop_words) :
    tokenized = tokenizer.tokenize(txt)
    statement_no_stop = [word.lower() for word in tokenized if word.lower() not in stop_words]
    lemmatized = [lemmatizer.lemmatize(token) for token in statement_no_stop]

    return lemmatized

def get_vectors_for_dictionary(source_text, all_dict_words) :
     tokenizer = RegexpTokenizer(r'\w+')
     stop_words = set(stopwords.words('english')) 
     lemmatizer = WordNetLemmatizer()

     tokens = get_tokens(source_text, tokenizer, lemmatizer, stop_words)
     similarity_matrix = []
     for dict in all_dict_words:
            dictionary = all_dict_words[dict]
            binary_vector = [1 if word in dictionary else 0 for word in tokens]
            similarity_matrix.append(binary_vector)
            binary_vector = np.array(similarity_matrix)
     
     return similarity_matrix
all_dict_words,ind =  get_dict_words("en-hi_acronym_dicts 2", "en", "hi")
source_text = get_txt("BTP_report.pdf")

g = get_vectors_for_dictionary(source_text, all_dict_words) #return vectors for all dictionary
budget = 10
ind_budget = [sum(lst) for lst in g]
sorted_sums = sorted(ind_budget, reverse=True)[:budget]  # get the top budget  maximum values in descending order
top_k_indices = [ind_budget.index(sum_val) for sum_val in sorted_sums] 
for entry in top_k_indices:
    print(ind[entry])








            

     