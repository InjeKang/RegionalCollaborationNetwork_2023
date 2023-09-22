from modules.GlobalVariables import *
from modules.preprocess import *

import pandas as pd
import swifter
from tqdm import trange
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager
from matplotlib import rc

"""
# check fonts
font_list = []
for i in range(len(fm.findSystemFonts(fontpaths="C:/Windows/Fonts", fontext="ttf"))):
    font_ = fm.findSystemFonts(fontpaths="C:/Windows/Fonts", fontext="ttf")[i].split("\\")[-1]
    font_list.append(font_)
print(sorted(font_list))
"""
# font directory
font_path = 'C:/Windows/Fonts/NGULIM.ttf'
fontprop = font_manager.FontProperties(fname=font_path, size=12)

# extract font's name
font_name = fm.FontProperties(fname=font_path).get_name()
# select font
plt.rcParams['font.family'] = font_name

from collections import defaultdict
import numbers
from collections import Counter


class NetworkAnalysis:
    def __init__(self, doc_type:str):
        self.doc_type = doc_type         

    def network_analysis_byType(self, type_, run): # type_ = collab OR knowledge
        if run:
            data = read_data(f"04.{self.doc_type}_collab.xlsx")            
            if self.doc_type == "paper":
                delimiter = ";"
                data.dropna(subset=["keywords", "affiliation_region"], inplace=True)
            elif self.doc_type == "patent":
                delimiter = "|"
                data.dropna(subset=["ipc", "applicant_region_cleansed"], inplace=True)

            if type_["collab"] and self.doc_type == "paper":
                network_column = "affiliation_region"
                sub_data = data[data["collab"].str.contains("collab")]
                result = self._network_analysis(sub_data, network_column, delimiter)
            if type_["collab"] and self.doc_type == "patent":
                network_column = "applicant_region_cleansed"
                sub_data = data[data["collab"].str.contains("collab")]
                result = self._network_analysis(sub_data, network_column, delimiter)
            if type_["knowledge"] and self.doc_type == "paper":
                network_column = "keywords"
                sub_data = data.copy()
                result = self._network_analysis(sub_data, network_column, delimiter)
            if type_["knowledge"] and self.doc_type == "patent":
                network_column = "ipc_4digit"
                sub_data = data.copy()
                result = self._network_analysis(sub_data, network_column, delimiter)
            return result

    def _network_analysis(self, data, network_column, delimiter):
            data[network_column] = data[network_column].str.lower()
            # edge list and adjacency matrix for all regions
            df_edgeList, df_adjacency_matrix = self._adjacency_matrix(data, network_column, delimiter)
            df_centrality = self._measure_centrality(data, network_column, delimiter)
            # edge list and adjacency matrix for each region
            region_Edgelist = []
            region_AdjMat = []
            region_Centrality = []
            for region in FilterData().region_list:
                if network_column == "applicant_region_cleansed" or network_column == "ipc_4digit":
                    sub_data_byRegion = data[data["applicant_region_cleansed"].str.startswith(region)]
                else:
                    sub_data_byRegion = data[data["affiliation_region"].str.startswith(region)]

                df_edgeList_by_region, df_adjacency_matrix_by_region = self._adjacency_matrix(sub_data_byRegion, network_column, delimiter)
                centrality_by_region = self._measure_centrality(sub_data_byRegion, network_column, delimiter)

                region_Edgelist.append(df_edgeList_by_region)
                region_AdjMat.append(df_adjacency_matrix_by_region)
                region_Centrality.append(centrality_by_region)

            # save the results
            with pd.ExcelWriter(f"data\\20.{self.doc_type}_{network_column}_Network_AdjMat.xlsx") as writer:
                df_adjacency_matrix.to_excel(writer, sheet_name="overall", index=False)
                for i in trange(len(FilterData().region_list)):
                    region_AdjMat[i].to_excel(writer, sheet_name = str(FilterData().region_list[i]), index=False)

            with pd.ExcelWriter(f"data\\20.{self.doc_type}_{network_column}_Network_EdgeList.xlsx") as writer:
                df_edgeList.to_excel(writer, sheet_name="overall", index=False)
                for i in trange(len(FilterData().region_list)):
                    region_Edgelist[i].to_excel(writer, sheet_name = str(FilterData().region_list[i]), index=False)              

            with  pd.ExcelWriter(f"data\\20.{self.doc_type}_{network_column}_Centrality.xlsx") as writer:
                df_centrality.to_excel(writer, sheet_name="overall", index=False)
                for i in trange(len(FilterData().region_list)):
                    region_Centrality[i].to_excel(writer, sheet_name = str(FilterData().region_list[i]), index=False)  

    def _adjacency_matrix(self, data, column_, delimiter):
        df_edgeList = self._edge_list(data, column_, delimiter)      
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)             
     
        # Get the unique values
        nodes = set(pd.concat([column_freq["column1"], column_freq["column2"]]))
        nodes = {x for x in nodes if x != ""}
        # Create an empty adjacency matrix with the same length as the node list
        to_adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
        # Iterate over the rows of the dataframe and update the adjacency matrix
        for row in column_freq.itertuples(index=False):
            col1, col2, freq = row
            to_adjacency_matrix.loc[col1, col2] += freq
            to_adjacency_matrix.loc[col2, col1] += freq
        return column_freq, to_adjacency_matrix

    def _edge_list(self, data, column_, delimiter):
        co_words_list = []
        data2 = data.copy()
        data2.dropna(subset=[column_], inplace=True)
        data2[column_] = data2[column_].str.split(delimiter)
        for co_word in data2[column_]:
            # get all combinations of co-words (pairs)  and reflect undirected network
            co_word_combinations = [tuple(sorted(pair)) for pair in itertools.combinations(co_word, 2)]
            co_words_list.append(co_word_combinations)
        co_words_list = list(itertools.chain.from_iterable(co_words_list))
        co_words_df = pd.DataFrame(co_words_list, columns=["column1", "column2"])
        return co_words_df        

    def _edgeList_to_dataframe(self, df_edgeList, column_):
        # column_freq = df_edgeList.groupby(['column1', 'column2']).size().reset_index(name="weight")
        # Measure network characteristics for an undirected network
        column_freq = df_edgeList.groupby(['column1', 'column2']).size().reset_index(name="freq")

        # Remove rows with empty strings
        filter_edgeList = column_freq.replace("", float("nan")).dropna(subset=["column1", "column2"], how="any")
        if column_ == "keywords" or column_ == "ipc_4digit":
            # filter data to reflect the error: at least one sheet must be visible when saving to an excel file
            filter_edgeList_subset = filter_edgeList[filter_edgeList["freq"] > 5]
            filter_edgeList_subset[["column1", "column2"]] = filter_edgeList_subset[["column1", "column2"]].apply(lambda x: x.str.strip())
            filter_keywords = FilterData()
            mask = filter_keywords.remove_unnecessary_keywords(filter_edgeList_subset, 2, "column1", "column2")
            filter_edgeList_subset = filter_edgeList_subset[~mask]
        else:
            filter_edgeList_subset = filter_edgeList.copy()     
        
        # Create dataframe for edges and nodes
        edges = filter_edgeList_subset.copy()
        nodes_df = pd.concat([edges["column1"], edges["column2"]]).value_counts().reset_index()
        nodes_df.columns = [column_, "freq"]
        return filter_edgeList_subset, nodes_df

    def _measure_centrality(self, data, column_, delimiter):
        # region_ = overall or by region // collab_ = collab or sole
        df_edgeList = self._edge_list(data, column_, delimiter)
        # convert edgelist to a dataframe
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)
        # identifying network structure
        node_freq = [tuple(row) for row in column_freq.to_records(index=False)]
        G = nx.Graph()
        G.add_weighted_edges_from(list(node_freq))    
        # measuring three types of centrality        
        degree_centrality_ = nx.degree_centrality(G)
        closeness_centrality_ = nx.closeness_centrality(G)
        betweenness_centrality_ = nx.betweenness_centrality(G)
        # constraints_ = nx.constraint(G)            
        # clustering_coefficient_ = nx.clustering(G)
        # convert the results into a dataframe
        toDF = pd.DataFrame({
            "degree": pd.Series(degree_centrality_),
            "closeness": pd.Series(closeness_centrality_),
            "betweenness": pd.Series(betweenness_centrality_),
            # "constraints": pd.Series(constraints_),
            # "clustering" : pd.Series(clustering_coefficient_)
        })
        # from index to a column
        toDF.reset_index(inplace=True)
        toDF = toDF.rename(columns = {"index" : column_})
        # toDF = toDF.assign(structural_holes = 2 - toDF["constraints"])
        # merge two dataframes - toDF and nodes_df
        mergedDF = pd.merge(toDF, nodes_df, on=column_, how = "inner")
        mergedDF = mergedDF[[column_, "freq", "degree", "closeness", "betweenness"]] # , "clustering", "constraints" , "structural_holes"
        mergedDF = mergedDF.apply(lambda x: x.round(3) if x.name in ["degree", "closeness", "betweenness"] else x) # round the values of the three columns only
        mergedDF = mergedDF.sort_values(by="freq", ascending=False)
        return mergedDF    

    def bipartite_network(self, run):
        if run:
            data = read_data(f"05.{self.doc_type}_only_selected.xlsx")
            if self.doc_type == "paper":
                delimiter = ";"
                main_area = "research_areas"
                knowledge_ = "keywords"
                region_ = "affiliation_region"
            elif self.doc_type == "patent":
                delimiter = "|"                
                main_area = "research_areas"
                knowledge_ = "ipc_4digit"
                region_ = "applicant_region_cleansed"
                data[main_area] = data[main_area].str.replace(RegEx().pattern_square_brackets_quotation, "", regex=True)
                data[main_area] = data[main_area].str.replace(RegEx().linebreaks, "", regex=True)
                data[main_area] = data[main_area].str.replace("\\n", "", regex=False)
            data[knowledge_] = data[knowledge_].str.lower()
            data[main_area] = data[main_area].str.replace(" ", "")            
            major_topics_byRegion = read_data(f"11.{self.doc_type}_trend_only_selected.xlsx", sheet_="topic20_overall")
            major_topics_byRegion[main_area] = major_topics_byRegion[main_area].str.replace("\n", "")
            major_topics_all = major_topics_byRegion.groupby(main_area, as_index=False)["trend"].sum()
            major_topics_all = major_topics_all.sort_values(by="trend", ascending=False)
            major_topics5 = major_topics_all[main_area].head(5)
            topic_list = []
            for major_topic in major_topics5:
                sub_data = data[data[main_area].str.contains(major_topic)]
                # bipartite
                df_edgeList = self._bipartite_network(sub_data, "key", region_, delimiter, knowledge_, delimiter)
                topic_list.append(df_edgeList)
            with pd.ExcelWriter(f"data\\21.{self.doc_type}_Bipartite.xlsx") as writer:
                for i in trange(len(topic_list)):
                    topic_list[i].to_excel(writer, sheet_name = str(major_topics5.iloc[i]), index=False)
    
    def _bipartite_network(self, data, key, column1, column1_split, column2, column2_split):
        df_edgeList, df_edgeList_withFreq, df_adjacency_matrix = self._bipartite_adjacency_matrix(data, key, column1, column1_split, column2, column2_split)
        return df_edgeList

    def _bipartite_filter_node(self, data, column_, delimiter):
        data[column_] = data[column_].str.split(delimiter)
        data[column_] = data[column_].apply(UnifyNames(doc_type="paper")._strip_row)
        filter_keywords = FilterData()
        mask = filter_keywords.remove_unnecessary_keywords(data, 1, column_, "column2") # data, no_, column1, column2
        filter_data = data[~mask]
        column_series = pd.Series([item for sublist in filter_data[column_] for item in sublist])
        column_list = set(column_series.unique())
        return filter_data, column_series, column_list

    def _bipartite_adjacency_matrix(self, data, key, column1, column1_split, column2, column2_split):
        G = nx.Graph()
        data.dropna(subset=[column1, column2], how="any", inplace=True)
        # Node for column 1 - affiliation_region
        data[column1] = data[column1].str.replace(" ", "").str.split(column1_split)
        column1_series = pd.Series([item for sublist in data[column1] for item in sublist])
        x_list = set(column1_series.unique())
        # Node for column 2 - keywords
        filter_data, colum2_series, y_list = self._bipartite_filter_node(data, column2, column2_split)
        G.add_nodes_from(x_list, bipartite=0)
        G.add_nodes_from(y_list, bipartite=1)

        # Dictionary to store edge frequencies
        edge_frequencies = {}

        # Iterate over the dataframe and add edges
        for idx, row in filter_data.iterrows():
            id_ = row[key]
            x_key = row[column1]
            y_key = row[column2]
            for x in x_key:
                for y in y_key:
                    edge = (x, y)
                    G.add_edge(*edge, key=id_)  # Using * to unpack the edge tuple

                    # Update edge frequency in the dictionary
                    edge_frequencies[edge] = edge_frequencies.get(edge, 0) + 1

        # Convert the bipartite graph into an edge list dataframe
        df_edgeList_withFreq = pd.DataFrame(G.edges(data=True), columns=[column1, column2, 'attributes'])
        df_edgeList_withFreq['attributes'] = df_edgeList_withFreq['attributes'].apply(lambda x: x['key'])

        # Add the edge frequencies to the edge list dataframe
        df_edgeList_withFreq['key_count'] = df_edgeList_withFreq.apply(lambda row: edge_frequencies[(row[column1], row[column2])], axis=1)
        # Reorder the columns
        new_order = [column1, column2, "key_count"]
        df_edgeList_withFreq = df_edgeList_withFreq.reindex(columns=new_order)
        # df_edgeList.sort_values(by=["affiliation_region", "keywords"])
        # df_edgeList_withFreq = df_edgeList.copy()
        # df_edgeList_withFreq['key_count'] = df_edgeList_withFreq.groupby('attributes')['attributes'].transform('count')
        # subset_df_edgeList = df_edgeList.drop_duplicates(subset='attributes', keep='first')
        # filter_edgeList = self._bipartite_filter_node(df_edgeList, column2, column2_split)
        # filter_edgeList = df_edgeList_withFreq[[column1, column2, "key_count"]]
        # # Remove rows with empty strings
        # filter_edgeList = filter_edgeList.replace("", float("nan")).dropna(subset=[column1, column2], how="any")   
        # Create an empty adjacency matrix with the same length as the node list
        to_adjacency_matrix = pd.DataFrame(0, index=x_list, columns=y_list)
        # Iterate over the rows of the dataframe and update the adjacency matrix
        for row in df_edgeList_withFreq.itertuples(index=False):
            col1, col2, freq = row
            to_adjacency_matrix.loc[col1, col2] += freq
        # Repeat rows based on the 'frequency' column
        df_edgeList = df_edgeList_withFreq.loc[df_edgeList_withFreq.index.repeat(df_edgeList_withFreq['key_count'])].reset_index(drop=True)
        df_edgeList = df_edgeList[df_edgeList["key_count"]>=5]

        return df_edgeList, df_edgeList_withFreq, to_adjacency_matrix    
                