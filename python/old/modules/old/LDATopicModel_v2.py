from modules.GlobalVariables import *

import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pickle

from konlpy.tag import Okt
tokenizer = Okt()


def lda_optimal_no(raw_paper_, raw_patent_, cleansed_paper_, cleansed_patent_):
    if raw_paper_:
        pass
    if raw_patent_:
        type_ = "patent"
        data = read_data("00.patent_merged_v3(with_corpus).xlsx", "Sheet1")
        # Preprocess
        data["corpus_cleansed"] = data.swifter.apply(lambda x: preprocess_kor(x["corpus"]), axis=1)

        data["corpus_cleansed"] = data["corpus"].apply(lambda x: ' '.join(
            [word for word in x.split() if tokenizer.pos(word)[0][1] == 'Noun']))
        
        # Use KoNLPy for data cleansing
        data["corpus_cleansed"] = data["corpus"].swfiter.apply(lambda x: ' '.join(tokenizer.nouns(x)) if pd.notnull(x) else '')

        data["corpus_cleansed"] = data["corpus_cleansed"].swfiter.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))                
        # Convert empty rows to NaN
        data["corpus_cleansed"].replace('', pd.NA, inplace=True)
        # Remove rows with NaN
        data.dropna(subset=["corpus_cleansed"], inplace=True)
        data.to_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
        # Create a dictionary and corpus for LDA model
        dictionary = Dictionary(data["corpus_cleansed"].swfiter.apply(lambda x: x.split()))
        # Filter out infrequent words (less than 10 occurrences)
        dictionary.filter_extremes(no_below=10)        
        corpus = [dictionary.doc2bow(text.split()) for text in data["corpus_cleansed"]]
        dictionary.save(f"data\\03.{type_}_LDA_dictionary") # cannot save dictionary or list with pickle
        with open(f"data\\03.{type_}_LDA_corpus", "wb") as f:
            pickle.dump(corpus, f)
        ####
        # # Checking keywords
        # word_frequencies = {dictionary[id]: freq for id, freq in dictionary.cfs.items()}
        # # Sort words by frequency in descending order
        # sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        # # Print the top 100 most frequent words
        # for word, frequency in sorted_words[:100]:
        #     print(f"{word}: {frequency}")      
        ####            

    if cleansed_patent_:
        type_ = "patent"
        data = pd.read_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
        # Load the dictionary 
        dictionary= Dictionary.load(f"data\\03.{type_}_LDA_dictionary")
        # Load the list of corpus from the pickle file
        with open("data/my_list.pkl", "rb") as f:
            corpus = pickle.load(f)

        #Coherence
        min_max = [1, 100, 1]
        model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, epoch = 30, texts= data["corpus_cleansed"], step =min_max[2], min_max = min_max)

        # Show graph for optimal number of topics        
        x = range(min_max[0], min_max[1], 5)
        plt.plot(x, coherence_values, marker = 'o')
        plt.xticks(x) #x-axis tick marks with integer
        plt.xlabel("Num Topics")
        plt.savefig(f"{type_}_optimal_no.png")
        plt.clf()
    

def lda_topicModeling(data_cleansed, paper_, patent_):
    if paper_:
        pass
    if patent_:
        data = data_cleansed.copy()
        # Load the dictionary 
        dictionary= Dictionary.load("data/03.{type_}_LDA_dictionary")
        # Load the list of corpus from the pickle file
        with open("data/my_list.pkl", "rb") as f:
            corpus = pickle.load(f)

        # Train the LDA model
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5)

        # Assign topics to each document
        data["topic_tags"] = [[topic[0] for topic in lda_model.get_document_topics(doc)] for doc in corpus]
        data.to_excel("data\\03.LDA_results.xlsx", index=False)
        return data


def compute_coherence_values(dictionary, corpus, epoch, texts, step, min_max: list):
    """ Compute c_v coherence
    Higher the topic coherence, the topic is more human interpretable
    If the coherence score seems to keep increasing,
    it may make better sense to pick the model that gave the highest CV before flattening out.    
    """
    coherence_values = []
    model_list = []
    for num_topics in range(min_max[0], min_max[1], step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=num_topics, 
                                           random_state=100,
                                           update_every=1, #how often the model parameters should be updated
                                           # chunksize=100, #number of documents to be used in each training
                                           passes=epoch, #total number of training passes, same as epochs
                                           iterations = 400, #how much loop for each document
                                           alpha='auto', #parameter for gamma function of Diriichlet distribution
                                           eta='auto', #parameter for gamma function of Diriichlet distribution
                                           per_word_topics=True #If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count)
                                           )
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        print(f"Num Topics: {num_topics}, Coherence Score: {coherencemodel.get_coherence()}")

    return model_list, coherence_values