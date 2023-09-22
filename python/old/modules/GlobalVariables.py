from modules.preprocess import *

import pandas as pd
import os
from os.path import join
import math
import multiprocessing
from multiprocessing import Pool, cpu_count
import numpy as np
from functools import partial
import itertools
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import KMeans
from networkx.algorithms import isomorphism
import ast



def read_data(filename, sheet_):
    default_path = os.getcwd()
    input_path = join(default_path, "data")        
    # change default directory to read data
    os.chdir(input_path)
    data = pd.read_excel(filename, engine="openpyxl", sheet_name = sheet_)
    os.chdir(default_path)
    return data


def readData_folder(paper_, patent_, run):
    if run:
        default_path = os.getcwd()
        if paper_:        
            input_path = join(default_path, "data\\paper")
            id_ = "UT (Unique WOS ID)"
            type_ = "paper"
        if patent_:
            input_path = join(default_path, "data\\patent")
            id_ = "WIPS ON key"        
            type_ = "patent"
        # read all excel files
        os.chdir(input_path)
        data = []
        for file in os.listdir(input_path):
            if file.endswith(".xlsx") or file.endswith(".xls"):
                file_path = os.path.join(input_path, file)
                df = pd.read_excel(file_path)
                data.append(df)
        # combine the excel files into a single dataframe and remove duplicate values
        combined_df = pd.concat(data, ignore_index=True)
        combined_df.drop_duplicates(subset=id_, keep="first", inplace=True)
        # save into an excel file
        os.chdir(default_path)
        combined_df.to_excel(f"data\\00.{type_}_merged.xlsx", index=False)

def preprocess_data(paper_, patent_, run):
    if run:
        if paper_:
            type_ = "paper"
            # read patent data
            data = read_data(f"00.{type_}_merged.xlsx", "Sheet1")
            # select relevant columns
            data2 = data[["UT (Unique WOS ID)", "Publication Year", "Article Title", "Abstract", "Author Keywords", "Authors", "Addresses"]]
            data2 = data2.rename(columns = {"UT (Unique WOS ID)":"key",
                                            "Publication Year": "year",
                                            "Article Title": "title",
                                            "Abstract": "abstract",
                                            "Authors": "authors", "Author Keywords":"keywords", "Addresses":"address"})
            # Splitting the addresses column
            data2[['affiliation', 'region']] = data2['address'].apply(lambda x: pd.Series(extract_affiliation_region(x)))
            data2["corpus"] = data2["title"].astype(str) + " " + data2["abstract"].astype(str)

            # Split the applicants by the separator and create a list of applicants
            applicants_list = data2["affiliation"].explode().tolist() # affiliation, region
            # Get the unique list of applicants and their frequencies
            applicants_freq = pd.Series(applicants_list).value_counts()
            # Sort the applicants based on their frequency in descending order
            sorted_applicants_freq = applicants_freq.sort_values(ascending=False)
            sorted_applicants_freq.to_excel("data\\99.authors_list.xlsx")

            data3 = data2[["key", "affiliation", "corpus", "year", "keywords", "region"]]
            data3.to_excel(f"data\\00.{type_}_merged_v3(with_corpus).xlsx", index=False)
            return data3
    
        if patent_:
            type_ = "patent"
            data = read_data(f"00.{type_}_merged.xlsx", "Sheet1")
            # select relevant columns
            data2 = data[["WIPS ON key", "등록일", "발명의 명칭", "요약", "Current IPC All", "출원인", "출원인 주소[KR]"]]
            data2 = data2.rename(columns = {"WIPS ON key":"key", 
                                            "발명의 명칭": "title",
                                            "요약": "abstract",
                                            "출원인": "applicants", "Current IPC All":"ipc", "출원인 주소[KR]":"address"})
            # modify regions
            lookup_table = read_data("00.look_up.xlsx", "Sheet1")
            data2["region"] = transform_region(data2["region"], lookup_table)
            # four-digit-IPC
            data2["ipc_7digit"] = data2.swifter.apply(lambda x: seven_digit(x["ipc"].split(" | ")), axis = 1)
            data2["ipc_4digit"] = data2.swifter.apply(lambda x: four_digit(x["ipc"].split(" | ")), axis = 1)
            # select year and convert from str to numeric
            data2["year"] = data2.swifter.apply(lambda x: get_year(x["등록일"]), axis=1)
            data2["year"] = pd.to_numeric(data2["year"], errors="coerce")
            data2["corpus"] = data2["title"] + " " + data2["abstract"]
            
            # unifying the applicants
            words_to_remove = read_data("00.words_to_remove.xlsx", "Sheet1")
            words_to_remove_list = words_to_remove["remove"].to_list()       
            data2["applicants_cleansed"] = data2["applicants"].apply(lambda x: ' | '.join(
                [remove_words(applicant, words_to_remove_list) for applicant in x.split(' | ')]))
                

            lookup_table = read_data("00.look_up.xlsx", "Sheet1")
            data2["applicants_cleansed"] = data2["applicants_cleansed"].replace(lookup_table.set_index("before")["after"])

            # # Split the applicants by the separator and create a list of applicants
            # applicants_list = data2["applicants_cleansed"].str.split(" \| ").tolist()
            # # Flatten the list of applicants
            # flat_applicants_list = [applicant for sublist in applicants_list for applicant in sublist]
            # # Get the unique list of applicants and their frequencies
            # applicants_freq = pd.Series(flat_applicants_list).value_counts()
            # # Sort the applicants based on their frequency in descending order
            # sorted_applicants_freq = applicants_freq.sort_values(ascending=False)

            data3 = data2[["key", "applicants", "applicants_cleansed", "corpus", "year", "ipc_4digit", "ipc_7digit", "region"]]
            data3.to_excel(f"data\\00.{type_}_merged_v3(with_corpus).xlsx", index=False)
            return data3
        

def edge_list(data, column_, column1_split):
    co_words_list = []
    data2 = data.copy()   
    if column_ == "ipc_7digit":
        data2[column_] = data2[column_].apply(ast.literal_eval)
        # data2[column_] = data2[column_].str.split(",")
    else:
        data2[column_] = data2[column_].str.split(column1_split)
    for co_word in data2[column_]:        
        co_words_list.extend(list(itertools.combinations(co_word, 2)))
       
    co_words_df = pd.DataFrame(co_words_list, columns = ["column1", "column2"])
    return co_words_df

def perform_clustering(G, no_):
    # Extract node features from the graph
    features = nx.to_numpy_matrix(G)
    features = np.asarray(features)  # Convert matrix to array

    # Perform clustering using KMeans algorithm
    kmeans = KMeans(n_clusters=no_)  # Specify the desired number of clusters
    labels = kmeans.fit_predict(features)

    return labels

def count_graphlets(G, graphlet):
    GM = isomorphism.GraphMatcher(G, graphlet)
    count = 0
    for subgraph in GM.subgraph_isomorphisms_iter():
        count += 1
    return count

