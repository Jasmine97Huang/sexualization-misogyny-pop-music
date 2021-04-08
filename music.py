import pandas as pd
import nltk
import string
from gensim import corpora, models
import collections
import re

# Some Functions from Last Time to get us started:
def get_wordnet_pos(word):
    '''
    Tags each word with its Part-of-speech indicator -- specifically used for lemmatization in the get_lemmas function
    '''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': nltk.corpus.wordnet.ADJ,
                'N': nltk.corpus.wordnet.NOUN,
                'V': nltk.corpus.wordnet.VERB,
                'R': nltk.corpus.wordnet.ADV}

    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def get_lemmas(text):
    '''
    Gets lemmas for a string input, excluding stop words, punctuation
    '''
    # Define stop words
    stop = nltk.corpus.stopwords.words('english') + list(string.punctuation)
    
    # Combine list elements together into a single string to use NLTK's tokenizer
    text = ' '.join(text)
    
    # tokenize + lemmatize words in text
    tokens = [i for i in nltk.word_tokenize(text.lower()) if i not in stop]
    lemmas = [nltk.stem.WordNetLemmatizer().lemmatize(t, get_wordnet_pos(t)) for t in tokens]
    return lemmas

def get_tokens(text):
    '''
    Gets all tokens (including stop words), excluding punctuation
    '''
    # drop punctuation, but keep stopwords for initial word counting
    text = text.translate(str.maketrans('', '', string.punctuation))

    # tokenize remaining words and make a list of them for input `text`
    tokens = [i for i in nltk.word_tokenize(text.lower())]
    return tokens

def social_connection(text, liwc_dict):
    '''
    Compute rel. percentage of LIWC 2007 'social' category:
    words like "mate," "talk," "child"
    '''
    
    liwc_counts = wordCount(text, liwc_dict)
    
    return liwc_counts[0]['social'] / liwc_counts[2]

def antisocial_perc(text, liwc_dict):
    '''
    Compute rel. percentage of LIWC 2007 'anger' and 'swear' categories:
    words like "kill," "hate," "annoyed," "damn," "fuck"
    '''
    liwc_counts = wordCount(text, liwc_dict)
    
    return (liwc_counts[0]['anger'] + liwc_counts[0]['swear']) / liwc_counts[2]

def positive_perc(text, liwc_dict):
    '''
    Compute rel. percentage of LIWC 2007 'posemo' categories:
    words like "love," "nice," "sweet"
    '''
    liwc_counts = wordCount(text, liwc_dict)
    
    return liwc_counts[0]['posemo'] / liwc_counts[2]

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    '''
    Computes Coherence values for LDA models with differing numbers of topics.
    
    Returns list of models along with their respective coherence values (pick
    models with the highest coherence)
    '''
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.ldamulticore.LdaMulticore(corpus=corpus,
                                                 id2word=dictionary,
                                                 num_topics=num_topics,
                                                 workers=8)
        model_list.append(model)
        coherence_model = models.coherencemodel.CoherenceModel(model=model, 
                                                               corpus=corpus,
                                                               dictionary=dictionary,
                                                               coherence='u_mass')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values

def fill_topic_weights(df_row, bow_corpus, ldamodel):
    '''
    Fill DataFrame rows with topic weights for topics in songs.
    
    Modifies DataFrame rows *in place*.
    '''
    try:
        for i in ldamodel[bow_corpus[df_row.name]]:
            df_row[str(i[0])] = i[1]
    except:
        return df_row
    return df_row

def top_songs_by_topic(music_df, ldamodel, corpus, ntop=1):
    '''
    Finds the top "n" songs by topic, which we can use for
    understanding the types of songs included in a topic.
    '''
    topn_songs_by_topic = {}
    for i in range(len(ldamodel.print_topics())):
        # For each topic, collect the most representative song(s)
        # (i.e. highest probability containing words belonging to topic):
        top = sorted(zip(range(len(corpus)), ldamodel[corpus]), 
                     reverse=True, 
                     key=lambda x: abs(dict(x[1]).get(i, 0.0)))
        topn_songs_by_topic[i] = [j[0] for j in top[:ntop]]
        
        # Print out the topn songs for each topic and return their indices as a dictionary for further analysis:
        print("Topic " + str(i))
        print(music_df[['title','year','artist']].loc[topn_songs_by_topic[i]])
        print("*******************************")
    
    return topn_songs_by_topic

'''
The following code is adapted from psyLex.
psyLex: an open-source implementation of the Linguistic Inquiry Word Count
Created by Sean C. Rife, Ph.D.
srife1@murraystate.edu // seanrife.com // @seanrife
Licensed under the MIT License
https://github.com/seanrife/psyLex
'''

def readDict(dictionaryPath):
    '''
    Function to read in an LIWC-style dictionary
    '''
    catList = collections.OrderedDict()
    catLocation = []
    wordList = {}
    finalDict = collections.OrderedDict()

    # Check to make sure the dictionary is properly formatted
    with open(dictionaryPath, "r") as dictionaryFile:
        for idx, item in enumerate(dictionaryFile):
            if "%" in item:
                catLocation.append(idx)
        if len(catLocation) > 2:
            # There are apparently more than two category sections; throw error and die
            sys.exit("Invalid dictionary format. Check the number/locations of the category delimiters (%).")

    # Read dictionary as lines
    with open(dictionaryPath, "r") as dictionaryFile:
        lines = dictionaryFile.readlines()

    # Within the category section of the dictionary file, grab the numbers associated with each category
    for line in lines[catLocation[0] + 1:catLocation[1]]:
        catList[re.split(r'\t+', line)[0]] = [re.split(r'\t+', line.rstrip())[1]]

    # Now move on to the words
    for idx, line in enumerate(lines[catLocation[1] + 1:]):
        # Get each line (row), and split it by tabs (\t)
        workingRow = re.split('\t', line.rstrip())
        wordList[workingRow[0]] = list(workingRow[1:])

    # Merge the category list and the word list
    for key, values in wordList.items():
        if not key in finalDict:
            finalDict[key] = []
        for catnum in values:
            workingValue = catList[catnum][0]
            finalDict[key].append(workingValue)
    return (finalDict, catList.values())

def wordCount(data, dictOutput):
    '''
    Function to count and categorize words based on an LIWC dictionary
    '''
    finalDict, catList = dictOutput
    
    # Create a new dictionary for the output
    outList = collections.OrderedDict()

    # Number of non-dictionary words
    nonDict = 0

    # Convert to lowercase
    data = data.lower()

    # Tokenize and create a frequency distribution
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(data)

    fdist = nltk.FreqDist(tokens)
    wc = len(tokens)

    # Using the Porter stemmer for wildcards, create a stemmed version of the data
    porter = nltk.PorterStemmer()
    stems = [porter.stem(word) for word in tokens]
    fdist_stem = nltk.FreqDist(stems)

    # Access categories and populate the output dictionary with keys
    for cat in catList:
        outList[cat[0]] = 0

    # Dictionaries are more useful
    fdist_dict = dict(fdist)
    fdist_stem_dict = dict(fdist_stem)

    # Number of classified words
    classified = 0

    for key in finalDict:
        if "*" in key and key[:-1] in fdist_stem_dict:
            classified = classified + fdist_stem_dict[key[:-1]]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_stem_dict[key[:-1]]
        elif key in fdist_dict:
            classified = classified + fdist_dict[key]
            for cat in finalDict[key]:
                outList[cat] = outList[cat] + fdist_dict[key]

    # Calculate the percentage of words classified
    if wc > 0:
        percClassified = (float(classified) / float(wc)) * 100
    else:
        percClassified = 0

    # Return the categories, the words used, the word count, 
    # the number of words classified, and the percentage of words classified.
    return [outList, tokens, wc, classified, percClassified]