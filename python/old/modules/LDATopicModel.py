from modules.GlobalVariables import *

import pandas as pd
import gensim
import pyLDAvis
import pyLDAvis.gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import pickle
from pprint import pprint

from konlpy.tag import Okt
tokenizer = Okt()


def lda_optimal_no(raw_paper_, raw_patent_, cleansed_paper_, cleansed_patent_, run):
    if run:
        if raw_paper_:
            type_ = "paper"
            data = read_data(f"00.{type_}_merged_v3(with_corpus).xlsx", "Sheet1")
            dictionary, corpus = preprocess_LDA(data, type_)
        if raw_patent_:
            type_ = "patent"
            data = read_data(f"00.{type_}_merged_v3(with_corpus).xlsx", "Sheet1")
            dictionary, corpus = preprocess_LDA(data, type_)     

        if cleansed_paper_:
            type_ = "paper"
            
        if cleansed_patent_:
            type_ = "patent"
            # Load the data with corpus clenased
            data = pd.read_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
            # Load the dictionary 
            dictionary= Dictionary.load(f"data\\03.{type_}_LDA_dictionary")
            # Load the list of corpus from the pickle file
            with open(f"data\\03.{type_}_LDA_corpus", "rb") as f:
                corpus = pickle.load(f)
            """
            # Checking keywords
            word_frequencies = {dictionary[id]: freq for id, freq in dictionary.cfs.items()}
            # Sort words by frequency in descending order
            sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
            # Print the top 100 most frequent words
            for word, frequency in sorted_words[:100]:
                print(f"{word}: {frequency}")      
            """      

            #Coherence
            min_max = [1, 100, 1]
            model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, epoch = 30, texts= data["corpus_cleansed"], step =min_max[2], min_max = min_max)

            # List of coherence values into a dataframe
            df_coherence = pd.DataFrame({'coherence': coherence_values})
            df_coherence.index += 1
            df_coherence.index.name = 'number'
            # Convert index to a column
            df_coherence.reset_index(level=0, inplace=True)
            df_coherence.to_excel(f"data\\03.{type_}_coherence_results.xlsx", index=False)



            # Show graph for optimal number of topics
            x = range(min_max[0], min_max[1], 1)
            plt.plot(x, coherence_values, marker = 'o')
            plt.xticks(x) #x-axis tick marks with integer
            plt.xlabel("Num Topics")
            plt.savefig(f"results\\{type_}_optimal_no.png")
            plt.clf()
    

def lda_topicModeling(opt_no, paper_, patent_, run):
    if run:
        if paper_:
            pass
        if patent_:
            type_ = "patent"
            data = pd.read_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
            # Load the dictionary 
            dictionary= Dictionary.load(f"data/03.{type_}_LDA_dictionary")
            # Load the list of corpus from the pickle file
            with open(f"data\\03.{type_}_LDA_corpus", "rb") as f:
                corpus = pickle.load(f)

            # Train the LDA model
            lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=opt_no, random_state=100, 
                                    update_every=1, #how often the model parameters should be updated
                                    # chunksize=100, #number of documents to be used in each training
                                    # passes=30, #total number of training passes, same as epochs
                                    iterations = 500, #how much loop for each document
                                    alpha='auto', #parameter for gamma function of Diriichlet distribution
                                    eta='auto', #parameter for gamma function of Diriichlet distribution
                                    per_word_topics=True #If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word, along with their phi values multiplied by the feature length (i.e. word count)
                                    )
            # results of the LDA model
            df_keywords = pd.DataFrame(lda_model.print_topics(num_words=20), columns=['no', 'keywords'])
            df_keywords["keywords"] = df_keywords["keywords"].str.replace("+", ",")
            df_keywords.to_excel(f"data\\03.{type_}_LDA_results_topicKeywords_topic{opt_no}.xlsx", index=False)
            vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
            pyLDAvis.save_html(vis, f'results\\{type_}_ldavis_topic{opt_no}.html') #saliency:topic에 특 단어가 두드러지게 많이 쓰였는지 / relevance: topic의 특정 단어가 나타날 확률의 가중평균
            
            # Assign topics to each document
            data["topic_tags"] = [[topic[0] for topic in lda_model.get_document_topics(doc)] for doc in corpus]
            # applicants re_cleansed
            df_applicants_cleansed = read_data(f"00.{type_}_merged_v3(with_corpus).xlsx", "Sheet1")
            data["applicants_cleansed"] = df_applicants_cleansed["applicants_cleansed"]
            data.to_excel(f"data\\03.{type_}_LDA_results_topic{opt_no}.xlsx", index=False)
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