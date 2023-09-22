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

    def collaboration_network(self, type_, run): # type_ = collab OR knowledge
        if run:
            if self.doc_type == "paper":
                data = read_data(f"04.{self.doc_type}_collab.xlsx")
                sub_data = data[data["collab"].str.contains("collab")]
                
                if type_["collab"]:
                    # edge list, adjacency matrix, and visualization
                    self._network_analysis_compilation(sub_data, "affiliation_region", ";")
                    # measuring centrality
                    result = self._measure_centrality(sub_data, "affiliation_region", ";", "overall")
                    for region in FilterData().region_list:
                        sub_data_byRegion = data[data["affiliation_region"].str.startswith(region)]
                        result = self._measure_centrality(sub_data_byRegion, "affiliation_region", ";", region)                   

                
                if type_["knowledge"]:
                    # edge list, adjacency matrix, and visualization                    
                    sub_data["keywords"] = sub_data["keywords"].str.lower()
                    self._network_analysis_compilation(sub_data, "keywords", ";")
                    # measuring centrality
                    result = self._measure_centrality(sub_data, "keywords", ";", "overall")
                    for region in FilterData().region_list:
                        sub_data_byRegion = data[data["affiliation_region"].str.startswith(region)]
                        result = self._measure_centrality(sub_data_byRegion, "keywords", ";", region)                    
                
                return result
                
    def _edgeList_to_dataframe(self, df_edgeList, column_):
        column_freq = df_edgeList.groupby(['column1', 'column2']).size().reset_index(name="weight")
        # Measure network characteristics for an undirected network
        column_freq = df_edgeList.groupby(['column1', 'column2']).size().reset_index(name="freq")
        # Create dataframe for edges and nodes
        edges = column_freq.copy()
        nodes_df = pd.concat([edges["column1"], edges["column2"]]).value_counts().reset_index()
        nodes_df.columns = [column_, "freq"]
        return column_freq, nodes_df

    def _measure_centrality(self, data, column_, column_split, type_): # type_ = overall or by region
        df_edgeList = self._edge_list(data, column_, column_split)
        # convert edgelist to a dataframe
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)
        # identifying network structure
        G = nx.Graph()
        G.add_weighted_edges_from(list(column_freq))    
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
        mergedDF = pd.merge(toDF, nodes_df, on=column_, how = "outer")
        mergedDF = mergedDF[[column_, "freq", "degree", "closeness", "betweenness"]] # , "clustering", "constraints" , "structural_holes"
        mergedDF = mergedDF.apply(lambda x: x.round(3) if x.name in ["degree", "closeness", "betweenness"] else x)
        mergedDF = mergedDF.sort_values(by="freq", ascending=False)
        mergedDF.to_excel(f"data\\07.NetworkResults_{column_}_{type_}.xlsx", index=False)
        return mergedDF    


    def _network_analysis_compilation(self, data, column_, column_split):
        # make edge list and adjacency matrix
        df_edgeList, df_adjacency_matrix = self._adjacency_matrix(data, column_, column_split)
        # self._visualize_network(df_adjacency_matrix, "overall")
        region_Edgelist = []
        region_AdjMat = []
        # make edge lists and adjacency matrixes for each region
        for region in FilterData().region_list:
            sub_data_byRegion = data[data[column_].str.startswith(region)]
            df_edgeList_by_region, df_adjacency_matrix_by_region = self._adjacency_matrix(sub_data_byRegion, column_, column_split)
            # self._visualize_network(df_adjacency_matrix_by_region, region)
            region_Edgelist.append(df_edgeList_by_region)
            region_AdjMat.append(df_adjacency_matrix_by_region)
        # df_region_EdgeList = pd.concat(region_Edgelist, ignore_index=True)
        # df_region_AdjMat = pd.concat(region_AdjMat, ignore_index=True)
        # save the results
        with pd.ExcelWriter(f"data\\06.{self.doc_type}_collabNetwork_AdjMat.xlsx") as writer:
            df_adjacency_matrix.to_excel(writer, sheet_name="overall", index=False)
            for region in trange(0, len(FilterData().region_list)):
                region_AdjMat[region].to_excel(writer, sheet_name = str(FilterData().region_list[region]), index=False)        
        
        with pd.ExcelWriter(f"data\\06.{self.doc_type}_collabNetwork_EdgeList.xlsx") as writer:
            df_edgeList.to_excel(writer, sheet_name="overall", index=False)
            for region in trange(0, len(FilterData().region_list)):
                region_Edgelist[region].to_excel(writer, sheet_name = str(FilterData().region_list[region]), index=False)

    def _adjacency_matrix(self, data, column_, column_split):
        df_edgeList = self._edge_list(data, column_, column_split)
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)
        
        df_edgeList["key"] = df_edgeList[["column1", "column2"]].apply(lambda row: " ".join(sorted(row)), axis=1)
        df_edgeList['key_count'] = df_edgeList.groupby('key')['key'].transform('count')
        subset_df_edgeList = df_edgeList.drop_duplicates(subset='key', keep='first')
        
        # if column_ == "affiliation_region":
        #     subset_df_edgeList = subset_df_edgeList[subset_df_edgeList["key_count"] >= 500]
        # else:
        #     pass
        #     subset_df_edgeList = subset_df_edgeList[subset_df_edgeList["key_count"] >= 500]

        filter_edgeList = subset_df_edgeList[["column1", "column2", "key_count"]]
        # To solve KeyError: (1) "" and (2) nan
        # Remove rows with empty strings
        filter_edgeList = filter_edgeList.replace("", float("nan")).dropna(subset=["column1", "column2"], how="any")   
        # Get the unique values
        nodes = set(data[column_].str.split(column_split, expand=True).stack().unique())
        nodes = {x for x in nodes if x != ""}
        # Create an empty adjacency matrix with the same length as the node list
        to_adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
        # Iterate over the rows of the dataframe and update the adjacency matrix
        for row in filter_edgeList.itertuples(index=False):
            col1, col2, freq = row
            to_adjacency_matrix.loc[col1, col2] += freq
            to_adjacency_matrix.loc[col2, col1] += freq
        return subset_df_edgeList, to_adjacency_matrix


    def _edge_list(self, data, column_, column_split):
        co_words_list = []
        data2 = data.copy()
        data2.dropna(subset=[column_], inplace=True)
        data2[column_] = data2[column_].str.split(column_split)
        for co_word in data2[column_]:        
            co_word_combinations = co_words_list.extend(list(itertools.combinations(co_word, 2)))
            co_words_list.extend(co_word_combinations)
            co_words_list.extend([(b, a) for a, b in co_word_combinations]) # to reflect undirected network
        co_words_df = pd.DataFrame(co_words_list, columns = ["column1", "column2"])
        return co_words_df        

     
    def _visualize_network(self, adjacency_matrix_, sample_): # sample_ = overall or by regions 
        G = nx.from_pandas_adjacency(adjacency_matrix_)
        groups_ = FilterData().group
        # Extract the group information from the node labels
        node_groups = []
        for node in G.nodes():
            group_found = False
            for group, attributes in groups_.items():
                if node.startswith(group):
                    node_groups.append(attributes)
                    group_found = True
                    break
            if not group_found:
                node_groups.append(groups_['etc'])
        # Extract the colors and shapes for each group
        node_colors = [group['color'] for group in node_groups]
        # Draw the network graph
        pos = nx.spring_layout(G)  # Position the nodes using a spring layout algorithm
        nx.draw_networkx(G, pos, with_labels=True, node_size = 100, node_color=node_colors, node_shape="o", font_family=fontprop.get_name())
        # Customize the plot appearance
        plt.title(f"Collaboration Network")
        plt.axis('off')    
        # Set the plot limits to focus on the part where nodes in groups are
        nodes_in_groups = [node for node, group in zip(G.nodes(), node_groups) if group != groups_['etc']]
        min_x = min(pos[node][0] for node in nodes_in_groups)
        max_x = max(pos[node][0] for node in nodes_in_groups)
        min_y = min(pos[node][1] for node in nodes_in_groups)
        max_y = max(pos[node][1] for node in nodes_in_groups)
        padding = 0.1  # Additional padding around the nodes
        plt.xlim(min_x - padding, max_x + padding)
        plt.ylim(min_y - padding, max_y + padding)
        # plt.show()
        plt.savefig(f"results\\CollabNetwork_{sample_}.png")
        plt.clf()   

