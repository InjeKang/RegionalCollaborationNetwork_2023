# from modules.preprocess import *

import pandas as pd
import os
from os.path import join
import math
import multiprocessing
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager

import numpy as np
from functools import partial
import itertools
from tqdm import tqdm
import networkx as nx
from sklearn.cluster import KMeans
from networkx.algorithms import isomorphism
import ast



class FilterData:
    def __init__(self):
        self.filter_columns_paper = ["UT (Unique WOS ID)", "Article Title", "Publication Year", "Abstract", "Author Keywords",
                                "WoS Categories", "Research Areas",
                                "Affiliations", "Addresses", "Times Cited, All Databases"]
        self.filter_columns_paper_renamed = ["key", "title", "year", "abstract", "keywords", "wos_categories", "research_areas",
                                        "affiliations", "addresses", "cited"]
        
        self.filter_columns_patent = ["WIPS ON key", "등록일", "발명의 명칭", "요약", "Current IPC All", "출원인", "출원인 주소[KR]"]
        self.filter_columns_patent_renamed = ["key", "date", "title", "abstract", "ipc", "applicants", "address"]

        self.region_list = ["경상북도", "경상남도", "부산", "대구", "울산"]
        self.group = {'경상남도': {'color': 'red'}, '경상북도': {'color': 'blue'}, '부산': {'color': 'green'}, '울산': {'color': 'orange'},
                      '대구': {'color': 'purple'}, 'etc': {'color': 'gray'},}
        
        
    def remove_unnecessary_keywords(self, data, no_, column1, column2):
        if no_ == 2: # when both column1 and column2 are used
            # Create a boolean mask for each condition
            mask_starts_with_ampersand = data[column1].str.startswith('&') | data[column2].str.startswith('&') # if a value starts with &
            mask_is_numeric = data[column1].str.isnumeric() | data[column2].str.isnumeric() # if a value is numeric
            mask_has_only_punctuation = data[column1].str.match(r'^\W+$') | data[column2].str.match(r'^\W+$') # if a value consists of only punctuations
            mask_length_1 = (data[column1].str.len() == 1) | (data[column2].str.len() == 1) # if the length of a value is 1
            mask_www = data[column1].str.startswith('www') | data[column2].str.startswith('www') # if a value starts with www
            # Combine all the masks using the logical OR (|) operator
            final_mask = mask_starts_with_ampersand | mask_is_numeric | mask_has_only_punctuation | mask_length_1
        elif no_ == 1: # when only the column1 is used
            # Create a boolean mask for each condition
            mask_starts_with_ampersand = data[column1].str.startswith('&')
            mask_is_numeric = data[column1].str.isnumeric()
            mask_has_only_punctuation = data[column1].str.match(r'^\W+$')
            mask_length_1 = (data[column1].str.len() == 1)
            mask_www = data[column1].str.startswith('www')
            # Combine all the masks using the logical OR (|) operator
            final_mask = mask_starts_with_ampersand | mask_is_numeric | mask_has_only_punctuation | mask_length_1            
        return final_mask


class RegEx:
     def __init__(self):
        # remove (1) values within parantheses, (2) parantheses, (3) whitespace before and after parantheses
        self.pattern_within_parentheses = r"\s*\([^)]*\)\s*"
        self.pattern_after_whitespace = r",\s+"

        self.pattern_select_square_brackets = r'\[|\]' # select square brackets
        self.pattern_square_brackets_quotation = r'[\[\]\'"]' # select square brackets and quotations(', ")
        self.pattern_punctuation = r'[^\w\s]' # select all punctuations
        self.pattern_splitting_authors = r'\[(.*?)\]' # selecting values in between square brackets
        self.pattern_splitting_affiliations = r'\s*;\s*(?=\[)'  # split authors' information >> [...]...; [...]...;
        self.pattern_excluding_except_us = r'^[a-zA-Z]{2}.*usa$'
        self.pattern_before_hyphen = r'(\w+)(?=_)'
        self.linebreaks = r'\r?\n'
        



class MultiProcess:
    def __init__(self):
        self=self

    def multi_process_delimiter(self, df, target_func, sheet_, delimiter_):
        n_cores = multiprocessing.cpu_count()-4
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        # output = pd.concat(pool.map(target_func, df_split))
        target_func_with_delimiter = partial(target_func, sheet_ = sheet_, delimiter=delimiter_)
        output = pd.concat(tqdm(pool.map(target_func_with_delimiter, df_split), total = n_cores))
        pool.close()
        pool.join()
        return output
    
    def multi_process(df, target_func, type_): # type_ = df or list
        n_cores = multiprocessing.cpu_count()-4
        if type_ ==  "df":        
            df_split = np.array_split(df, n_cores)
            pool = Pool(n_cores)
            output = pd.concat(tqdm(pool.map(target_func, df_split)))
        else:
            list_ = []        
            list_split = np.array_split(df, n_cores)
            pool = Pool(n_cores)
            output = list_.append(pool.map(target_func, list_split))
        pool.close()
        pool.join()
        return output
    
    def multi_process_split_designated(self, data, func, split_func):
        n_cores = multiprocessing.cpu_count()-4
        # n_cores = 6
        df_split = self._split_dataframe(data, split_func, n_cores)
        pool = Pool(n_cores)
        # df = pd.concat(pool.map(func, df_split))
        df = pd.concat(tqdm(pool.map(func, df_split), total = n_cores))
        pool.close()
        pool.join()
        return df

    def _split_dataframe(self, data, split_func, num_partitions):
        df_list = split_func(data)
        df_split = [df_list[i::num_partitions] for i in range(num_partitions)]
        return df_split


def read_data(filename, sheet_="Sheet1"):
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
        # Load all excel files in the folder and remove duplicate values
        df_merged = read_all_files(input_path, id_)
        # save into an excel file
        os.chdir(default_path)
        df_merged.to_excel(f"data\\00.{type_}_merged.xlsx", index=False)


def read_all_files(path_, id_):
        os.chdir(path_)
        data = []
        for file in os.listdir(path_):
            if file.endswith(".xlsx") or file.endswith(".xls"):
                file_path = os.path.join(path_, file)
                df = pd.read_excel(file_path)
                data.append(df)
        # combine the excel files into a single dataframe and remove duplicate values
        df_merged = pd.concat(data, ignore_index=True)
        df_merged.drop_duplicates(subset=id_, keep="first", inplace=True)
        return df_merged