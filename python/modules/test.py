from modules.GlobalVariables import *
from modules.preprocess import *

import pandas as pd
import numpy as np
import re

# def _transform_region(self, data):
#     lookup_table = read_data("99.look_up.xlsx", sheet_=f"{self.doc_type}_pplicant_region_map")

def _mapLookup_multi(data, delimiter):
    data["applicant_region_cleansed"] = data.apply(lambda x: _mapLookup(x["applicant_region"], delimiter) 
                                                   if x["collab"] == "collab" else x["applicant_region"], axis=1)
    return data        

def _mapLookup(data, delimiter):
    lookup_table = read_data("99.look_up.xlsx", sheet_=f"patent_applicant_region_map")
    list_ = []  # Provide a default value for the 'list_' variable to solve UnboundLocalError: local variable 'list_' referenced before assignment
    if isinstance(data, str) and data.startswith('['):
        data = re.sub(RegEx().pattern_square_brackets_quotation , "", data)
        list_ = data.split(delimiter)
    elif isinstance(data, str):
        list_ = data.split(delimiter)
    elif isinstance(data, list):
        list_ = data
    x_list2 = [x.strip() for x in list_ if x.strip() != ""] # remove unnecessary whitespaces
    unified_list = []
    for element in x_list2:
        if isinstance(element, (int, float)):
            pass
        else:
            lookup = lookup_table[lookup_table['before'] == element]['after']
            if not lookup.empty:
                unified_list.append(lookup.iloc[0])
            else:
                unified_list.append(element)
    # to make sure all elements in unified_list are strings
    unified_list = [x.strip() for x in unified_list if x.strip() != ""]
    unified_list = [str(x) for x in unified_list]
    return delimiter.join(unified_list)
    

def test():
    data = read_data("03.patent_unifyNames.xlsx")  
    data.dropna(subset=["ipc", "year", "applicant_region"], inplace=True)
    data["collab"] = data["applicant_region"].swifter.apply(lambda x: Descriptive(doc_type="patent")._determine_collab(x,  "|"))
    # select regions from address
    target_func_mapLookup= _mapLookup_multi
    data = MultiProcess().multi_process_delimiter(data, target_func_mapLookup, "|")    
    # data["address"].iloc[55000:55050]
    # data2 = data[data["address"].str.strip() != ""]    
    data.to_excel("data\\03.patent_unifyNames_v2(regionClassification).xlsx", index=False)
    return data





# def test():
#     data = {
#         "affiliation_region": [
#             "ulsan_X;ulsan_Y",
#             "seoul_X;ulsan_Y",
#             "busan_X;ulsan_Y",
#             "busan_X;daejeon_Y",
#             "daejeon_X;ulsan_Y",
#         ]
#     }

#     df = pd.DataFrame(data)

#     # Define the list of elements to check for the start of the affiliation_region
#     regions_to_check = ["busan", "ulsan"]

#     # # Use the apply method to split the strings by the delimiter ";" and check if all elements start with any of the regions
#     # selected_rows = df[df["affiliation_region"].apply(lambda x: all(affil.startswith(region) for region in regions_to_check for affil in x.split(";")))]

#     # selected_rows = df[df["affiliation_region"].str.split(";").apply(lambda x: all(affil.startswith(region) for region in regions_to_check for affil in x))]
#     # selected_rows = df[df["affiliation_region"].apply(lambda x: all(x.startswith(region) for region in regions_to_check))]

#     # pattern = r"^(?=.*(?:{}))".format("|".join(regions_to_check))
#     # selected_rows = df[df["affiliation_region"].str.contains(pattern, regex=True)]

#     # selected_rows = df[df["affiliation_region"].apply(lambda x:
#     #                                                   all(affil.strip().startswith(region) for region in regions_to_check
#     #                                                       for affil in x.split(";")))]
#     def select_only_sample(data, delimiter, regions_to_check):
#         split_data = data.split(delimiter)
#         test_regions = [region for region in split_data if any(region.startswith(reg) for reg in regions_to_check)]
#         test_result = [str(region in test_regions) for region in split_data]
#         return ";".join(list(set(test_result)))

#     df["only_region"] = df["affiliation_region"].apply(lambda x: select_only_sample(x, ";", regions_to_check))




