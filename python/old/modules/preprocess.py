from modules.GlobalVariables import *
# from modules.GlobalVariables import read_data


import pandas as pd
import numpy as np
import re
import pickle
import swifter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from konlpy.tag import Okt
tokenizer = Okt()
   



def remove_words(applicant, words_to_remove_list):    
    revised_applicant = applicant
    for word in words_to_remove_list:
        revised_applicant = revised_applicant.replace(word, "")
    revised_applicant = revised_applicant.strip()  # Remove leading/trailing whitespace
    return revised_applicant

def preprocess_LDA(data, type_):
    # Preprocess
    if type_ == "patent":
        data["corpus_cleansed"] = data.swifter.apply(lambda x: preprocess_kor(x["corpus"]), axis=1)
    else: # type_ == "paper"
        data["corpus_cleansed"] = data.swifter.apply(lambda x: preprocess_eng(x["corpus"]), axis=1)
    # Convert empty rows to NaN
    data["corpus_cleansed"].replace('', pd.NA, inplace=True)
    # Remove rows with NaN
    data.dropna(subset=["corpus_cleansed"], inplace=True)
    data.to_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
    # Create a dictionary and corpus for LDA model
    word_lists = data["corpus_cleansed"].tolist()
    dictionary = Dictionary(word_lists)
    # Filter out infrequent words
    dictionary.filter_extremes(no_below=10, keep_tokens=[tokenid for tokenid, freq in dictionary.cfs.items()])
    corpus = [dictionary.doc2bow(text) for text in word_lists] 
    dictionary.save(f"data\\03.{type_}_LDA_dictionary") # cannot save dictionary or list with pickle
    with open(f"data\\03.{type_}_LDA_corpus", "wb") as f:
        pickle.dump(corpus, f)
    return dictionary, corpus


def preprocess_kor(data):
    if data =="":
        return np.nan
    else:
        y  = " ".join([word for word in data.split() if tokenizer.pos(word)[0][1] == 'Noun'])
        x = tokenizer.nouns(y)
        # exclude stopwords
        stop_words = ["본", "발명"]
        keywords = [w for w in x if not w in stop_words]
        keywords = [i for i in keywords if len(i) >=2]
        return keywords



def four_digit(data):
    four_digitIPC = [x[0:4] for x in data]
    return four_digitIPC
    
def seven_digit(data):
    seven_digitIPC = [x[0:8] for x in data]
    return seven_digitIPC

def get_year(data):    
    return data[0:4]



def transform_region(region, lookup_table, delimiter):
    transformed_regions = []
    regions = region.split(delimiter)
    for subregion in regions:
        lookup = lookup_table[lookup_table['before'] == subregion]['after']
        if not lookup.empty:
            transformed_regions.append(lookup.iloc[0])
        else:
            transformed_regions.append(subregion)
    return delimiter.join(transformed_regions)


def preprocess_eng(data):
    if data == "":
        return np.nan
    else:
        stop_words = set(stopwords.words('english'))
        stop_words.update(['https', 'also', 'new', 'et', 'al', 'could', 'use', 'one', 'may', 'well', 'main', 'x', 'might', 'within',
                    'paper', 'us', 'however', 'even', 'framework', 'need', 'based', 'first', 'would', 'pp', 'study',
                    'three', 'research', 'fig', 'meaning', 'work', 'analysis', 'question', 'terms', 'key', 'design', 'example',
                    'possible', 'model', 'among', 'particular', 'j', 'important', 'two', 'see', 'way', 'general', 'many', 'process',
                    'control', 'number', 'large', 'available', "using", "results", "p", "used", "data", "method", "compared"])
        x = word_tokenize(data) #split into words
        for j in range(len(x)):
            x[j] = x[j].lower() #to lowercases
        words = [word for word in x if word.isalpha()] #remove punctuation
        # lancaster=LancasterStemmer()
        # words = [lancaster.stem(word) for word in words] #stemming or lemmatizing
        # words = [p_stemmer.stem(word) for word in words]
        # words = [wnl.lemmatize(word) for word in words]
        keywords = [w for w in words if not w in stop_words] #stopwords # keywords = [w for w in words if not w in stop_words and len(w) > 1]
        # keywords = [i for i in keywords if len(i) > 1]
        return keywords


def extract_affiliation_region(address):
    if pd.isna(address):  # Check for missing or NaN values
        return '', ''
    authors = address.split('; [')
    affiliations = []
    regions = []
    if len(authors) > 1:
        for author in authors:
            author_parts = author.split(']')        
            try:
                author_no = author_parts[0].count(";")
                # Affiliation
                affiliation_parts = author_parts[1].strip().split(",")[0]
                # Region
                """
                #1 Daegu 42471, North Gyeongsan, South Korea // #2 Daegu 42471, South Korea

                """
                region_parts = author_parts[1].strip().split(",")[-3:-1]
                region_parts = ",".join(region_parts)
                region_parts_cleansed = re.sub(r'\s+|\d+', '', region_parts)



                if author_no == 0:
                    affiliations.append([affiliation_parts])
                    regions.append([region_parts])
                else:
                    affiliations.append([affiliation_parts] * author_no)
                    regions.append([region_parts_cleansed]*author_no)
        
            except:
                author_info = author.split(",")
                affiliations.append([author_info[0]])
                region_parts = author_info[-3:-1]
                region_parts = ",".join(region_parts)
                region_parts_cleansed = re.sub(r'\s+|\d+', '', region_parts)
                regions.append([region_parts_cleansed])

        

        affiliations_flatten = [item for sublist in affiliations for item in sublist]
        regions_flatten = [item for sublist in regions for item in sublist]

        return affiliations_flatten, regions_flatten
    
    else:
        return np.nan, np.nan