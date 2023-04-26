import pdfplumber
import time
import numpy as np
# Import your set cover functions
from setcover import select_glossaries1
from probabilitysetcover import select_glossaries2
from facilityCover import select_glossaries3
from concaveOverModular import select_glossaries4

# Mixture of submodular components
def Fw(S, w, functions, src_lang, trans_lang, glossaries_path):
    return sum(w[i] * f(S, src_lang, trans_lang, glossaries_path) for i, f in enumerate(functions.values()))


# Jaccard loss function
def jaccard_loss(S, T, k):
    max_jaccard = [max(len(Gamma(s).intersection(Gamma(t))) / len(Gamma(s).union(Gamma(t))) for t in T) for s in S]
    return 1 - (1 / k) * sum(max_jaccard)

# Stochastic gradient descent for learning parameters
def learn_parameters(S, Y, ground_truth_summaries, lambda_, initial_weights, functions, max_iter=1000, learning_rate=0.01):
    weights = np.array(initial_weights)
    N = len(ground_truth_summaries)

    for _ in range(max_iter):
        gradient = np.zeros_like(weights)

        for i in range(N):
            S_i = ground_truth_summaries[i]

            max_loss = float('-inf')
            max_S0 = None

            for S0 in Y:
                if S0 == S_i:
                    continue
                
                loss = Fw(S0, weights, functions) + jaccard_loss(S0, S_i, len(S0))
                if loss > max_loss:
                    max_loss = loss
                    max_S0 = S0
            
            gradient += Fw(max_S0, weights, functions, "en", "hi", "en-hi_acronym_dicts_2") - Fw(S_i, weights, functions, "en", "hi", "en-hi_acronym_dicts_2")


        gradient /= N
        gradient += lambda_ * weights
        weights -= learning_rate * gradient

        # Project weights back to the feasible region
        weights = np.maximum(0, weights)
        weights /= np.sum(weights)

    return weights

def Gamma(s):
    pass

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

def get_preference_order(pdf_path, ocr_lang, src_lang, trans_lang, glossaries_path):
    if src_lang != "en" or ocr_lang != "en":
        return None

    if isMachineReadable(pdf_path):
        # Prepare the submodular functions dictionary
        submodular_functions = {
            'set_cover': select_glossaries1,
            'probability_set_cover': select_glossaries2,
            'facility_cover': select_glossaries3,
            'concave_over_modular': select_glossaries4
        }

        # Replace the following placeholders with your specific data
        S = [...]  # ground-truth summaries
        Y = [...]  # structured output space
        initial_weights = [0.25, 0.25, 0.25, 0.25]  # initial weights for submodular components
        lambda_ = 0.1  # regularization parameter

        # Learn weights for the submodular functions
        weights = learn_parameters(S, Y, S, lambda_, initial_weights, submodular_functions)

        # Compute the best glossary using the learned weights
        best_glossary_score = float('-inf')
        best_glossary = None

        for glossary in glossaries_path:
            score = Fw(glossary, weights, submodular_functions)
            if score > best_glossary_score:
                best_glossary_score = score
                best_glossary = glossary

        return best_glossary

    return None

start = time.time()
print(get_preference_order("pdf24_merged.pdf", "en", "en", "hi", "en-hi_acronym_dicts_2"))
end = time.time()
print(end - start)




