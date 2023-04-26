def get_coverage(dict_words, tokens, tokenizer, lemmatizer, stop_words) :
    dict_words = np.array(dict_words)
    coverage = []

    for term in tokens :
        tokenized = [lemmatizer.lemmatize(word.lower()) for word in tokenizer.tokenize(term) if word.lower() not in stop_words]
        if len(tokenized) == 0 :
            coverage.append(0)
        try :
            matched_inds = np.where(dict_words == tokenized)[0]
        except :
            continue
        

        count = len(list(matched_inds))
        prob = count/len(tokens)
        coverage.append(prob)   

    return (coverage)