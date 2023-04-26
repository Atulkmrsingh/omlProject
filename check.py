def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words) :
    coverage = [] #stores probability of every token tokens are concepts.
    for word in tokens : #for every word in pdf calculating the probability of being present in a dictionary
        tokenized = [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(word) if word.lower() not in stop_words]
        if len(tokenized) == 0 :
            coverage.append(0)
            continue
        try :
            matched_inds = np.where(dict_words == tokenized)
            count = len(matched_inds)
            prob = count/len(tokens)
            coverage.append(prob)
        except :
            coverage.append(0)
            continue   
    return np.array(coverage)

    def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words) :
    tokens = np.array(tokens)
    coverage = []

    for term in dict_words :
        tokenized = [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(term) if word.lower() not in stop_words]
        if len(tokenized) == 0 :
            continue
        try :
            matched_inds = np.where(tokens == tokenized[0])[0]
        except :
            continue
        
        if len(tokenized) == 1 :
            coverage += list(matched_inds)
        
        else :
            for ind in matched_inds :
                discard = False
                for j in range(1, len(tokenized)) :
                    if tokens[min(ind + j, len(tokens) - 1)] != tokenized[j] :
                        discard = True
                        break
                if not discard :
                    coverage += list(range(ind, ind + len(tokenized)))  

    return set(coverage)