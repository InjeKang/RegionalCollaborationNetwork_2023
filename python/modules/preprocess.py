from modules.GlobalVariables import *



import pandas as pd
import numpy as np
import re
import pickle
import swifter
import math
from functools import partial
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from konlpy.tag import Okt
tokenizer = Okt()
   


class PreprocessFilter:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type
    
    def filter_column(self, run):
        if run:
            data = read_data(f"00.{self.doc_type}_merged.xlsx", sheet_="Sheet1")
            if self.doc_type == "paper":
                filter_data = data[FilterData().filter_columns_paper]
                filter_data.columns = FilterData().filter_columns_paper_renamed
                filter_data["year"] = filter_data["year"].astype(int)
            if self.doc_type == "patent":
                filter_data = data[FilterData().filter_columns_patent]
                filter_data.columns = FilterData().filter_columns_patent_renamed
                # four (or seven)-digit-IPC
                filter_data["ipc_7digit"] = filter_data["ipc"].swifter.apply(lambda x: self._seven_digit(x, "|"))
                filter_data["ipc_4digit"] = filter_data["ipc"].swifter.apply(lambda x: self._four_digit(x, "|"))
                # select year and convert from str to numeric
                filter_data["year"] = filter_data.swifter.apply(lambda x: self._get_year(x["date"]), axis=1)
                filter_data["year"] = pd.to_numeric(filter_data["year"], errors="coerce")

            filter_data.to_excel(f"data\\01.{self.doc_type}_filterColumn.xlsx", index=False)

            return filter_data
    
    def _get_year(self, data):
        return data[0:4]

    def _four_digit(self, data, delimiter):
        data_splitted = data.split(delimiter)
        four_digitIPC = [x.strip()[0:4] for x in data_splitted]
        return delimiter.join(four_digitIPC)
        
    def _seven_digit(self, data, delimiter):
        data_splitted = data.split(delimiter)
        seven_digitIPC = [x[0:8] for x in data_splitted]
        return delimiter.join(seven_digitIPC)

    def _select_only_sample(self, data, delimiter, select_sample):
        split_data = data.split(delimiter)
        test_regions = [x for x in split_data if any(x.startswith(reg) for reg in select_sample)]
        only_region_selected = []
        for element in test_regions:
            only_region = re.findall(RegEx().pattern_before_hyphen, element)
            only_region_selected.extend(only_region)
            
        test_result = [str(x in test_regions) for x in split_data]
        return [delimiter.join(list(set(test_result))), delimiter.join(list(set(only_region_selected)))]

class UnifyNames:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type

    def unify_affiliation(self, run):
        if run:
            if self.doc_type == "paper":
                data = read_data(f"01.{self.doc_type}_filterColumn.xlsx", sheet_="Sheet1")
                data2 = data.dropna(subset=["affiliations"])
                data2["affiliations"] = data2["affiliations"].str.lower()
                # data2 = data2.iloc[0:50]            
                # unify affiliations' names by cleansing
                data2["affiliation_cleansed"] = data2["affiliations"].swifter.apply(lambda x: self._cleanse_name(x, ";"))
                # # unify affilations' names by mapping
                # data2["affiliation_cleansed"] = data2["affiliation_cleansed"].swifter.apply(lambda x: self._mapLookup(x, ";"))
                data2.to_excel(f"data\\03.{self.doc_type}_unifyNames.xlsx", index=False)

            if self.doc_type == "patent":
                data = read_data(f"01.{self.doc_type}_filterColumn.xlsx", sheet_="Sheet1")
                data.dropna(subset=["ipc", "address", "applicants"], inplace=True)
                # test = data.iloc[0:500]
                # unify applicants' names by cleansing
                target_func_cleansing = self._cleanse_name_multi
                data = MultiProcess().multi_process_delimiter(data, target_func_cleansing, f"{self.doc_type}_map", "|")
                # test["applicants_cleansed"] = test["applicants"].swifter.apply(lambda x: self._cleanse_name(x, "|"))
                # select regions from address
                target_func_mapLookup= self._mapLookup_multi
                data = MultiProcess().multi_process_delimiter(data, target_func_mapLookup, f"{self.doc_type}_map", "|")
                # test["region"] = test["address"].swifter.apply(lambda x: self._mapLookup(x, "|"))
                # match applicants with regions
                target_func_matchRegion= self._matchRegion_multi
                data = MultiProcess().multi_process_delimiter(data, target_func_matchRegion, f"{self.doc_type}_map", "|")                
                # test["applianct_region"] = test.apply(lambda x: self.match_region(x["applicants_cleansed"], x["region"], "|"), axis=1)
                # check the affiliations' names
                self._unique_list(data, "applicants_cleansed", "|")
                data.to_excel(f"data\\03.{self.doc_type}_unifyNames.xlsx", index=False)
                return data
    
    def _cleanse_name_multi(self, data, delimiter):
        if self.doc_type == "paper":
            column_ = "affiliations"
            column_cleansed = "affiliation_cleansed"
        elif self.doc_type == "patent":
            column_ = "applicants"
            column_cleansed = "applicants_cleansed"
        data[column_cleansed] = data.apply(lambda x: self._cleanse_name(x[column_], delimiter), axis=1)
        return data
    
    def _cleanse_name(self, data, delimiter):
        words_to_remove_ = read_data("99.look_up.xlsx", sheet_=f"{self.doc_type}_exclusion")
        words_to_remove = words_to_remove_["exclusion"].astype(str).tolist()        
        list_ = []  # Provide a default value for the 'list_' variable to solve UnboundLocalError: local variable 'list_' referenced before assignment
        if isinstance(data, str) and data.startswith('['):
            data = re.sub(RegEx().pattern_square_brackets_quotation , "", data)
            list_ = data.split(delimiter)
        elif isinstance(data, str):
            list_ = data.split(delimiter)
        x_list2 = [x.strip() for x in list_] # remove unnecessary whitespaces
        unified_list = []
        for element in x_list2:    
            # Remove words within parantheses
            element_cleansed = re.sub(RegEx().pattern_within_parentheses, "", element)
            # Remove whitespace after a comma, ex) test, co >> test,co
            element_cleansed = re.sub(RegEx().pattern_after_whitespace, " ", element_cleansed)
            # Remove commas and periods
            element_cleansed = re.sub(r"[.,]"," ", element_cleansed)
            # Remove unnecessary words such as corporation, llc, inc, etc.
            element_cleansed = ' '.join([word for word in element_cleansed.split() if word not in words_to_remove])
            element_cleansed.strip()
            unified_list.append(element_cleansed)
        return delimiter.join(unified_list)

    def _mapLookup_multi(self, data, sheet_, delimiter):
        if self.doc_type == "paper":
            column_ = "affiliation_cleansed"
            column_result = "affiliation_region"
        elif self.doc_type == "patent" and sheet_ == f"{self.doc_type}_map" :
            column_ = "address"
            column_result = "region"
        elif self.doc_type == "patent" and sheet_ == "ipc_ksic" :
            column_ = "ipc_4digit"
            column_result = "research_areas"
        data[column_result] = data.apply(lambda x: self._mapLookup(x[column_], sheet_, delimiter), axis=1)
        # data["applicant_region_cleansed"] = data.apply(lambda x: self._mapLookup(x["applicant_region"], delimiter) 
        #                                         if x["collab"] == "collab" else x["applicant_region"], axis=1)
        return data        

    def _mapLookup(self, data, sheet_, delimiter):
        lookup_table = read_data("99.look_up.xlsx", sheet_=sheet_)
        if sheet_ == "ipc_ksic":
            lookup_table["after"] = lookup_table["after"].str.replace("\n", " ")
            lookup_table["before"] = lookup_table["before"].str.replace(" ", "")
            lookup_table["before"] = lookup_table["before"].apply(lambda x: PreprocessFilter(self.doc_type)._four_digit(x, ","))
            lookup_table["before"] = lookup_table["before"].str.split(",")
            lookup_table = lookup_table.explode("before")            
        list_ = []  # Provide a default value for the 'list_' variable to solve UnboundLocalError: local variable 'list_' referenced before assignment
        if isinstance(data, str) and data.startswith('['):
            data = re.sub(RegEx().pattern_square_brackets_quotation , "", data)
            list_ = data.split(delimiter)
        elif isinstance(data, str):
            list_ = data.split(delimiter)
        elif isinstance(data, list):
            list_ = data
        x_list2 = [x.strip() for x in list_] # remove unnecessary whitespaces
        unified_list = []
        for element in x_list2:
            if self.doc_type == "patent":
                element = re.sub(RegEx().pattern_punctuation, "", element)
                element = element.split(" ")[0]
            lookup = lookup_table[lookup_table['before'] == element]['after']
            if not lookup.empty:
                unified_list.append(lookup.iloc[0])
            else:
                if sheet_ == "ipc_ksic":
                    pass
                else:
                    unified_list.append(element)
        # to make sure all elements in unified_list are strings
        unified_list = [x.strip() for x in unified_list if x.strip() != ""]
        unified_list = [str(x) for x in unified_list]                
        return delimiter.join(unified_list)

    def _unique_list(self, data, column_, delimiter):
        affiliations_list = data[column_].str.split(delimiter).tolist()
        # Flatten the list of affiliations
        flat_affiliations_list = [applicant for sublist in affiliations_list for applicant in sublist]
        flat_affiliations_list = [x.strip() for x in flat_affiliations_list]
        # Get the unique list of affiliations and their frequencies
        affiliations_freq = pd.Series(flat_affiliations_list).value_counts()
        # Sort the affiliations based on their frequency in descending order
        sorted_affiliations_freq = affiliations_freq.sort_values(ascending=False)
        df_sorted_affiliations_freq = pd.DataFrame({column_: sorted_affiliations_freq.index, 'freq': sorted_affiliations_freq.values})
        df_sorted_affiliations_freq.to_excel(f"data\\99.{self.doc_type}_subset_aff_freq.xlsx", index=False)

    def _matchRegion_multi(self, data, delimiter):
        if self.doc_type == "paper":
            pass
        elif self.doc_type == "patent":
            data["applicant_region"] = data.apply(lambda x: self.match_region(x["applicants_cleansed"], x["region"], delimiter), axis=1)
        return data        


    def match_region(self, column1_, column2_, delimiter):
        if self.doc_type == "paper":
            pass
            # data = read_data(f"03.{self.doc_type}_unifyNames.xlsx", sheet_="Sheet1")
            # target_func = self._mapLookup_multi
            # data2 = MultiProcess().multi_process_delimiter(data, target_func, ";")
            # data["affiliation_region"] = data["affiliation_cleansed"].swifter.apply(lambda x: self._mapLookup(x, ";"))
            # data2.to_excel(f"data\\03.{self.doc_type}_unifyNames_v2(regionClassification).xlsx", index=False)
            # return data2
        elif self.doc_type == "patent": # x["address"], x["applicants_cleansed", "region"], "|"
            applicants_split = column1_.split(delimiter)
            regions_split = column2_.split(delimiter)
            result = delimiter.join([f'{region}_{applicant}' for applicant, region in zip(applicants_split, regions_split)])
            return result



    def _strip_row(self, data):
        if isinstance(data, float):
            return data  # Return the float as is, assuming it's not meant to be iterated.
        return [item.strip() for item in data]   

class Descriptive:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type

    def descriptive_(self, type_, run): # type = collab or overall
        if run:
            # data = read_data(f"03.{self.doc_type}_unifyNames_v2(regionClassification).xlsx")
            data = read_data(f"04.{self.doc_type}_collab.xlsx")
            if self.doc_type == "paper":                
                delimiter = ";"
                column_affiliation_region = "affiliation_region"
                data = data[data["year"]<2023]
            elif self.doc_type == "patent":
                delimiter = "|"
                column_affiliation_region = "applicant_region_cleansed"
                data = data[data["year"]<2023]
                # data.drop('ipc_4digit', axis=1, inplace=True)
                # data["ipc_4digit"] = data["ipc"].swifter.apply(lambda x: PreprocessFilter(self.doc_type)._four_digit(x, "|"))
            #     target_func = UnifyNames(self.doc_type)._mapLookup_multi
            #     data = MultiProcess().multi_process_delimiter(data, target_func, "ipc_ksic", delimiter)
            #     data["research_areas"] = UnifyNames(self.doc_type)._mapLookup(self, data, "ipc_ksic", delimiter)
            # # data["collab"] = data[column_affiliation_region].swifter.apply(lambda x: self._determine_collab(x, delimiter))   
            # data.to_excel(f"data\\04.{self.doc_type}_collab_v2.xlsx", index=False)                

            # trend analysis
            year_range = data["year"].unique()
            if type_["collab"]:
                sub_data = data[data["collab"].str.contains("collab")]
                annual_trend_collab = self._descriptive_annual_trend(sub_data, column_affiliation_region, year_range, "collab", delimiter)
                trend_by_topic_collab, trend_by_topic20_collab = self._descriptive_by_topic(sub_data, column_affiliation_region, "all", "collab", delimiter)

            if type_["overall"]:
                annual_trend_overall = self._descriptive_annual_trend(data, column_affiliation_region, year_range, "overall", delimiter)
                trend_by_topic_overall, trend_by_topic20_overall = self._descriptive_by_topic(data, column_affiliation_region, "all", "overall", delimiter)
            
            with pd.ExcelWriter(f"data\\10.{self.doc_type}_trend.xlsx") as writer:
                annual_trend_collab.to_excel(writer, sheet_name="annual_collab", index=False)
                trend_by_topic_collab.to_excel(writer, sheet_name="topic_collab", index=False)       
                trend_by_topic20_collab.to_excel(writer, sheet_name="topic20_collab", index=False)
                annual_trend_overall.to_excel(writer, sheet_name="annua_overall", index=False)
                trend_by_topic_overall.to_excel(writer, sheet_name="topic_overall", index=False)
                trend_by_topic20_overall.to_excel(writer, sheet_name="topic20_overall", index=False)                                


            if type_["only_selected"]:
                data[["only_region", "unit_region"]] = data.swifter.apply(lambda x: 
                                                                            PreprocessFilter(doc_type=self.doc_type)._select_only_sample(x[column_affiliation_region], delimiter, FilterData().region_list)
                                                                            , axis=1, result_type="expand")
                if delimiter == "|": # because | is treated as a special character in regular expression
                    sub_data = data[(data["only_region"] == "True") & (data["unit_region"].str.contains(delimiter, regex=False))]
                    sub_data["collab_count"] = sub_data["unit_region"].str.count("\|")
                else:
                    sub_data = data[(data["only_region"] == "True") & (data["unit_region"].str.contains(delimiter))]
                    sub_data["collab_count"] = sub_data["unit_region"].str.count(delimiter)

                sub_data.to_excel(f"data\\05.{self.doc_type}_only_selected.xlsx")               
                
                # sub_data["collab_count"].min()
                # sub_data["collab_count"].value_counts()
                sub_data2 = sub_data.copy()
                trend_topic_overall, trend_topic20_overall = self._descriptive_by_topic(sub_data2, column_affiliation_region, "all", "only_selected", delimiter)
                sub_data["unit_region"] = sub_data["unit_region"].apply(lambda row: delimiter.join(sorted(row.split(delimiter))))
                region_list = sub_data["unit_region"].unique()
                
                # test = pd.DataFrame({"sample": region_list})
                # test["sample"] = test["sample"].apply(lambda row: delimiter.join(sorted(row.split(delimiter))))
                # test["collab_count"] = test["sample"].str.count(delimiter)
                # for i in range(1, 4):
                #     sub_test = test[test["collab_count"] == i]
                #     print(len(sub_test["sample"].unique()))
                # test["collab_count"].value_counts()
                trend_annual_df = []
                trend_topic_df = []
                trend_topic20_df = []
                for region in region_list:
                    sub_data_byRegion = sub_data[sub_data["unit_region"] == region]                    
                    for i in range(sub_data_byRegion["collab_count"].min(), sub_data_byRegion["collab_count"].max()+1):
                        version_ = str(i)+"_"+region
                        sub_data_byCount = sub_data_byRegion[sub_data_byRegion["collab_count"] == i]
                        annual_trend = self._descriptive_annual_trend(sub_data_byCount, column_affiliation_region, year_range, "only_selected", delimiter)
                        trend_topic_byRegion, trend_topic20_byRegion = self._descriptive_by_topic(sub_data_byCount, column_affiliation_region, region, "only_selected", delimiter)
                        annual_trend["version"] = version_
                        trend_topic_byRegion["version"] = version_
                        trend_topic20_byRegion["version"] = version_

                        trend_annual_df.append(annual_trend)
                        trend_topic_df.append(trend_topic_byRegion)
                        trend_topic20_df.append(trend_topic20_byRegion)

                    df_annual = pd.concat(trend_annual_df, ignore_index=True)
                    df_toipc = pd.concat(trend_topic_df, ignore_index=True)             
                    df_toipc20 = pd.concat(trend_topic20_df, ignore_index=True)
                
                with pd.ExcelWriter(f"data\\11.{self.doc_type}_trend_only_selected.xlsx") as writer:
                    trend_topic_overall.to_excel(writer, sheet_name="topic_overall", index=False)
                    trend_topic20_overall.to_excel(writer, sheet_name="topic20_overall", index=False)       
                    df_annual.to_excel(writer, sheet_name="annual_byRegion", index=False)
                    df_toipc.to_excel(writer, sheet_name="topic_byRegion", index=False)
                    df_toipc20.to_excel(writer, sheet_name="topic20_byRegion", index=False)   

                return data

    def _determine_collab(self, data, delimiter):
        if isinstance(data, str):
            if delimiter in data:
                return "collab"
            else: 
                return "sole"
        elif isinstance(data, str):
            if len(data) > 1:
                return "collab"
            else: 
                return "sole"
            
    def _descriptive_annual_trend(self, data, column_affiliation_region, year_range, type_, delimiter):
        trend_ = []
        if type_ == "only_selected":
            data["unit_region"] = data["unit_region"].str.split(delimiter)
            unit_region_series = pd.Series([item for sublist in data["unit_region"] for item in sublist])
            region_list = unit_region_series.unique()
            for region in region_list:
                sub_data = data[data[column_affiliation_region].str.contains(region)]
                annual_trend = sub_data.groupby("year").size().reindex(year_range, fill_value=0).reset_index(name="trend")
                # annual_trend.columns = ["year", "trend"]
                annual_trend["region"] = region
                annual_trend = annual_trend.sort_values(by='year', ascending=True)
                trend_.append(annual_trend)
            merged_trend = pd.concat(trend_, ignore_index=True)           
        else:
            for region in FilterData().region_list:
                sub_data = data[data[column_affiliation_region].str.contains(region)]
                annual_trend = sub_data.groupby("year").size().reindex(year_range, fill_value=0).reset_index(name="trend")
                # annual_trend.columns = ["year", "trend"]
                annual_trend["region"] = region
                annual_trend = annual_trend.sort_values(by='year', ascending=True)
                trend_.append(annual_trend)
            merged_trend = pd.concat(trend_, ignore_index=True)
        return merged_trend
                
    def _descriptive_by_topic(self, data, column_affiliation_region, sample_, type_, delimiter):        
        # data2 = data.copy()
        data.dropna(subset=['research_areas'], inplace=True)
        if isinstance(data["research_areas"].iloc[0], str):
            data["research_areas"] = data["research_areas"].str.replace(" ", "").str.split(delimiter)
        else: # when rows are in list..already cleansed once
            pass
        research_areas_series = pd.Series([item for sublist in data["research_areas"] for item in sublist])
        research_area_list = research_areas_series.unique()
        trend_total = []
        trend_top20 = []
        if type_ == "only_selected" and sample_ != "all":
            # data.dropna(subset=["unit_region"], inplace=True)
            region_list = sample_.split(delimiter)
            for region in region_list:
                sub_data = data[data[column_affiliation_region].str.contains(region)]
                research_area_counts = sub_data["research_areas"].explode().value_counts()
                research_area_counts = research_area_counts.reindex(research_area_list, fill_value=0).reset_index()
                research_area_counts.columns = ["research_areas", "trend"]
                research_area_counts["region"] = region
                # all topics
                sorted_research_areas = research_area_counts.sort_values(by="research_areas", ascending=False)            
                trend_total.append(sorted_research_areas)
                # top 20 topics
                top_20_research_areas = research_area_counts.sort_values(by="trend", ascending=False)
                top_20_research_areas = top_20_research_areas.head(20)
                trend_top20.append(top_20_research_areas)               
        else:
            for region in FilterData().region_list:
                sub_data = data[data[column_affiliation_region].str.contains(region)]
                research_area_counts = sub_data["research_areas"].explode().value_counts()
                research_area_counts = research_area_counts.reindex(research_area_list, fill_value=0).reset_index()
                research_area_counts.columns = ["research_areas", "trend"]
                research_area_counts["region"] = region
                # all topics
                sorted_research_areas = research_area_counts.sort_values(by="research_areas", ascending=False)            
                trend_total.append(sorted_research_areas)
                # top 20 topics
                top_20_research_areas = research_area_counts.sort_values(by="trend", ascending=False)
                top_20_research_areas = top_20_research_areas.head(20)
                trend_top20.append(top_20_research_areas)
            
        total_research_areas = pd.concat(trend_total, ignore_index=True)
        top20_reseaach_areas = pd.concat(trend_top20, ignore_index=True)
        return total_research_areas, top20_reseaach_areas