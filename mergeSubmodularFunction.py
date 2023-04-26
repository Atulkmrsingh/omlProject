import numpy as np
import pdfplumber
from setcover import select_glossaries1
from probabilitysetcover import select_glossaries2
from facilityCover import select_glossaries3
from concaveOverModular import select_glossaries4
import time

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

def normalize_function(f, V):
    f_V = f(V)
    return lambda S: {k: v / f_V[k] for k, v in f(S).items()}

def mixture_function(weights, functions):
    return lambda S: {k: sum(w * f(S)[k] for w, f in zip(weights, functions)) for k in S}

def greedy_optimization(Fw, V, budget):
    S = {}
    for _ in range(budget):
        best_element = None
        best_gain = -np.inf
        for k, v in V.items():
            if k not in S:
                Fw_S = Fw(S)
                Fw_S_with_element = Fw({**S, k: v})
                gain = sum(Fw_S_with_element[k] - Fw_S[k] for k in S)
                if gain > best_gain:
                    best_gain = gain
                    best_element = k
        S[best_element] = V[best_element]
    return S

# Returns preference order for english machine readable source, None otherwise
def get_preference_order(pdf_path, ocr_lang, src_lang, trans_lang, glossaries_path):
    if src_lang != "en" or ocr_lang != "en":
        return None

    if isMachineReadable(pdf_path):
        # Collect glossary dictionaries from submodular functions
        V1 = select_glossaries1(pdf_path, src_lang, trans_lang, glossaries_path)
        V2 = select_glossaries2(pdf_path, src_lang, trans_lang, glossaries_path)
        V3 = select_glossaries3(pdf_path, src_lang, trans_lang, glossaries_path)
        V4 = select_glossaries4(pdf_path, src_lang, trans_lang, glossaries_path)
        count=0
        dV={}
        for i in V1.split(','):
            dV[i]=11-count
            count=count+1
        count=0
        for i in V2.split(','):
            dV[i]=11-count
            count=count+1
        count=0
        for i in V3.split(','):
            dV[i]+=11-count
            count=count+1
        count=0
        for i in V4.split(','):
            dV[i]+=11-count
            count=count+1
        sorted_dict = dict(sorted(dV.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        # Combine dictionaries from submodular functions
        # V = {**dV3, **dV4}

        # functions = [normalize_function(f, V) for f in [select_glossaries1, select_glossaries2, select_glossaries3, select_glossaries4]]
        # weights = [0.25, 0.25, 0.25, 0.25]  # Example weights, adjust them as needed
        # Fw = mixture_function(weights, functions)

        # # Optimize the combined function
        # budget = 3  # Adjust the budget as needed
        # solution = greedy_optimization(Fw, V, budget)

        # return solution

    return None

start = time.time()
print(get_preference_order("pdf24_merged.pdf", "en", "en", "hi", "en-hi_acronym_dicts_2"))
end = time.time()
print(end-start)
