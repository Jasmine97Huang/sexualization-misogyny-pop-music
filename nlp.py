import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import string
from gensim import corpora, models

# Define stop words + punctuation + study-specific stop-words
stop = nltk.corpus.stopwords.words('english') + list(string.punctuation) + ["amp", "39", "subscribe", "follow",
                                                                            "link", "ermenegildo", "zegna", "uomo",
                                                                            "music", "applause", "um", "facebook"
                                                                           ]

def pos_tag(text):
    # Tokenize words using nltk.word_tokenize, keeping only those tokens that do not appear in the stop words we defined
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in stop]
    
    # Label parts of speech automatically using NLTK
    pos_tagged = nltk.pos_tag(tokens)
    return pos_tagged

def plot_top_adj(series, data_description):
    # Apply part of Speech tagger that we wrote above to any Pandas series that pass into the function
    pos_tagged = series.apply(pos_tag)

    # Extend list so that it contains all words/parts of speech for all the captions
    pos_tagged_full = []
    for i in pos_tagged:
        pos_tagged_full.extend(i)
    
    # Create Frequency Distribution of different adjectives and plot the distribution
    fd = nltk.FreqDist(word + "/" + tag for (word, tag) in pos_tagged_full if tag[:2] == 'JJ')
    fd.plot(15, title='Top 15 Adjectives for ' + data_description);
    return

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in stop]
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return lemmas

def plot_top_lemmas(series, data_description, n = 20):
    lemmas = series.apply(get_lemmas)

    # Extend list so that it contains all words/parts of speech for all the captions
    lemmas_full = []
    for i in lemmas:
        lemmas_full.extend(i)

    nltk.FreqDist(lemmas_full).plot(n, title='Top 10 Lemmas Overall for ' + data_description);
    return

def plot_top_tfidf(series, data_description, n = 15):
    # Get lemmas for each row in the input Series
    lemmas = series.apply(get_lemmas)
    
    # Initialize Series of lemmas as Gensim Dictionary for further processing
    dictionary = corpora.Dictionary([i for i in lemmas])

    # Convert dictionary into bag of words format: list of (token_id, token_count) tuples
    bow_corpus = [dictionary.doc2bow(text) for text in lemmas]
    
    # Calculate TFIDF based on bag of words counts for each token and return weights:
    tfidf = models.TfidfModel(bow_corpus)
    tfidf_weights = {}
    for doc in tfidf[bow_corpus]:
        for ID, freq in doc:
            tfidf_weights[dictionary[ID]] = np.around(freq, decimals = 2)

    # highest TF-IDF values:
    top_n = pd.Series(tfidf_weights).nlargest(n)
    
    # Plot the top 10 weighted words:
    plt.plot(top_n.index, top_n.values)
    plt.xticks(rotation='vertical')
    plt.title('Top {} Lemmas (TFIDF) for '.format(n) + data_description);
    
    return