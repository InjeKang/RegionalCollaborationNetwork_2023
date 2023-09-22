from modules.GlobalVariables import *
from modules.preprocess import *

import pandas as pd
import swifter
from tqdm import trange
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 폰트 경로 설정
font_path = 'C:/Windows/Fonts/NGULIM.ttf'
# 폰트 이름 추출
font_name = fm.FontProperties(fname=font_path).get_name()
# 폰트 설정
plt.rcParams['font.family'] = font_name

from collections import defaultdict
import numbers
from collections import Counter



def collaboration_network(sample_, opt_no, paper_, patent_, run):
    if run:
        if paper_:
            type_ = "paper"
        if patent_:
            type_ = "patent"
            data = read_data(f"03.{type_}_LDA_results_topic{opt_no}.xlsx", "Sheet1")
            data2 = data.dropna(how='any')
            data2['applicants_cleansed'] = data2['applicants_cleansed'].str.replace(' ', '')
            data2['region'] = data2['region'].str.replace(' ', '')            
            # unify the names of regions
            data2 = unify_region(data2, 'region', "|")            
            data2["applicant_region"] = data2.swifter.apply(lambda x :
                                                            applicants_with_region(x["applicants_cleansed"], x["region"], "|"), axis=1)
            data2["topic_tags"] = data2["topic_tags"].apply(eval)
            # Make an edge list and adjacency matrix
            region_list = ["경상북도", "경상남도", "부산", "대구", "울산",
                           "서울", "경기도", "대전", "충청북도", "충청남도", "전라남도", "전라북도", "세종"]
            adjacency_matrix_ = []
            edgeList_ = []
            for topic_ in trange(0, opt_no):
                sub_data = data2[data2["topic_tags"].apply(lambda x: topic_ in x)]
                # From data, make edge list and adjacency matrix
                df_edgeList, adjacency_matrix_by_topic = adjacency_matrix(sub_data, sample_, "|")                
                df_edgeList = df_edgeList.assign(topic = topic_)
                edgeList_.append(df_edgeList)
                visualize_network(adjacency_matrix_by_topic, topic_, sample_)
                if sample_ == "region":
                    sub_matrix = adjacency_matrix_by_topic.loc[region_list, region_list]
                if sample_ == "applicant_region":
                    sub_matrix = adjacency_matrix_by_topic[
                        adjacency_matrix_by_topic.columns.str.endswith(tuple(region_list)) |
                        adjacency_matrix_by_topic.index.str.endswith(tuple(region_list))]
                adjacency_matrix_.append(sub_matrix)
            df_merged = pd.concat(edgeList_, ignore_index=True)
            df_merged.to_excel(f"data\\07.{type_}_collabNetwork_edgeList_{sample_}_topic{opt_no}.xlsx", index=False)           
            if sample_ == "region":
                with pd.ExcelWriter(f"data\\07.{type_}_collabNetwork_adjMat_{sample_}_topic{opt_no}.xlsx") as writer:
                    for topic_ in trange(0, opt_no):
                        adjacency_matrix_[topic_].to_excel(writer, sheet_name = str(topic_), index=False)
            return adjacency_matrix_

def unify_region(data, sample_, delimiter):
    lookup_table = read_data("00.look_up.xlsx", "Sheet1")
    # Remove punctuations excepet the delimiter, "|"
    data[sample_] = data[sample_].apply(lambda x: re.sub(r'[^\w\s|]', '', x))
    data[sample_] = data[sample_].str.replace("특별시", "")
    data[sample_] = data[sample_].str.replace("특별자치도", "")
    data[sample_] = data[sample_].str.replace("특별자치시", "")
    data[sample_] = data[sample_].str.replace("광역시", "")
    data[sample_] = data[sample_].swifter.apply(lambda x: transform_region(x, lookup_table, delimiter))

    # data[sample_].str.split("|", expand=True).stack().unique()
    return data


def visualize_network(adjacency_matrix_, topic_, sample_):
    G = nx.from_pandas_adjacency(adjacency_matrix_) 
    groups = {
        '경상남도': {'color': 'red'},
        '경상북도': {'color': 'blue'},
        '부산': {'color': 'green'},
        '울산': {'color': 'orange'},
        '대구': {'color': 'purple'},
        'etc': {'color': 'gray'},
    }    
    if sample_ == "region":
        # Extract the group information from the node labels
        node_groups = [groups.get(node, groups['etc']) for node in G.nodes()]
        # Extract the colors and shapes for each group
        node_colors = [group['color'] for group in node_groups]
        # node_shapes = [group['shape'] for group in node_groups]
    if sample_ == "applicant_region":
        # Extract the group information from the node labels
        node_groups = []
        for node in G.nodes():
            group_found = False
            for group, attributes in groups.items():
                if node.endswith(group):
                    node_groups.append(attributes)
                    group_found = True
                    break
            if not group_found:
                node_groups.append(groups['etc'])
        # Extract the colors and shapes for each group
        node_colors = [group['color'] for group in node_groups]
        # node_shapes = [group['shape'] for group in node_groups]

    # Draw the network graph
    pos = nx.spring_layout(G)  # Position the nodes using a spring layout algorithm
    nx.draw_networkx(G, pos, node_size = 100, node_color=node_colors, node_shape="o", with_labels=False)

    # Customize the plot appearance
    plt.title(f"Network Structure of Topic {topic_}")
    plt.axis('off')    
    # Set the plot limits to focus on the part where nodes in groups are
    nodes_in_groups = [node for node, group in zip(G.nodes(), node_groups) if group != groups['etc']]
    min_x = min(pos[node][0] for node in nodes_in_groups)
    max_x = max(pos[node][0] for node in nodes_in_groups)
    min_y = min(pos[node][1] for node in nodes_in_groups)
    max_y = max(pos[node][1] for node in nodes_in_groups)
    padding = 0.1  # Additional padding around the nodes
    plt.xlim(min_x - padding, max_x + padding)
    plt.ylim(min_y - padding, max_y + padding)
    plt.savefig(f"results\\CollabNetwork_{sample_}_{topic_}.png")
    plt.clf()        



def adjacency_matrix(data, column_, column1_split):
    df_edgeList = edge_list(data, column_, column1_split)
    df_edgeList["key"] = df_edgeList[["column1", "column2"]].apply(lambda row: " ".join(sorted(row)), axis=1)
    df_edgeList['key_count'] = df_edgeList.groupby('key')['key'].transform('count')
    subset_df_edgeList = df_edgeList.drop_duplicates(subset='key', keep='first')
    filter_edgeList = subset_df_edgeList[["column1", "column2", "key_count"]]
    # To solve KeyError: (1) "" and (2) nan
    # Remove rows with empty strings
    filter_edgeList = filter_edgeList.replace("", float("nan")).dropna(subset=["column1", "column2"], how="any")   
    # Get the unique values
    nodes = set(data[column_].str.split(column1_split, expand=True).stack().unique())
    nodes = {x for x in nodes if x != ""}
    # Create an empty adjacency matrix with the same length as the node list
    to_adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    # Iterate over the rows of the dataframe and update the adjacency matrix
    for row in filter_edgeList.itertuples(index=False):
        col1, col2, freq = row
        to_adjacency_matrix.loc[col1, col2] += freq
        to_adjacency_matrix.loc[col2, col1] += freq
    return subset_df_edgeList, to_adjacency_matrix



def biPartite_network(opt_no, column_, paper_, patent_, run):
    if run:
        if paper_:
            type_ = "paper"
            data = read_data(f"03.{type_}_LDA_results_topic{opt_no}.xlsx")
        if patent_:
            type_ = "patent"
            data = read_data(f"03.{type_}_LDA_results_topic{opt_no}.xlsx", "Sheet1")            
            data2 = data.dropna(how='any')
            if column_ == ["ipc_4digit", "topic_tags"]:
                df_edgeList, adjacency_matrix_by_topic = adjacency_matrix_bipartite(data, "key", column_[0], ",", column_[1], ",")
                df_edgeList["rank"] = df_edgeList.groupby("topic_tags")["key_count"].rank(method='dense', ascending=False)                
                subset_edgeList = df_edgeList[df_edgeList["rank"] <= 5]
                subset_edgeList = subset_edgeList.sort_values(by=["topic_tags", "rank"], ascending=True)
                with pd.ExcelWriter(f"data\\07.{type_}_bipartiteNetwork_edgeList_topic{opt_no}.xlsx") as writer:
                    df_edgeList.to_excel(writer, sheet_name = "overall", index=False)
                    subset_edgeList.to_excel(writer, sheet_name = "top5", index=False)                
                adjacency_matrix_by_topic.to_csv(f"data\\07.{type_}_bipartiteNetwork_adjMat_topic{opt_no}.csv", index=True)
            if column_ == ["applicants_cleansed", "region"]:
                data2['applicants_cleansed'] = data2['applicants_cleansed'].str.replace(' ', '')
                data2['region'] = data2['region'].str.replace(' ', '')
                data2["applicant_region"] = data2.swifter.apply(lambda x :
                                                                applicants_with_region(x["applicants_cleansed"], x["region"], "|"), axis=1)
                region_list = ["경상북도", "경상남도", "부산", "대구", "울산"]
                df_frequency_byRegion = [] # Major topics of each region
                for region in region_list:                
                    sub_data = data2[data2["region"].str.contains(region, na=False)]
                    sub_data_collab = sub_data[sub_data["applicants"].str.contains("|", na=False)]
                    # Frequency of each topic
                    freq_by_region = frequency(sub_data_collab['topic_tags'], region)
                    df_frequency_byRegion.append(freq_by_region)
                    
                    # Bipartite Network in a dataframe
                    bipartite_edgelist = adjacency_matrix_bipartite(sub_data, "key", "applicant_region", "|", "topic_tags", ",")
                    bipartite_edgelist.to_excel(f"data\\04.{type_}_{region}_biNet.xlsx", index=False)
                
                # Major topics of each region
                df_frequency_total = pd.concat(df_frequency_byRegion, axis=0)
                df_frequency_total.to_excel(f"data\\05.{type_}_majorTopics.xlsx", index=False)
                return df_frequency_total


def applicants_with_region(applicants, regions, delimiter):    
    applicants_split = applicants.split(delimiter)
    regions_split = regions.split(delimiter)
    merged = delimiter.join([f'{applicant}_{region}' for applicant, region in zip(applicants_split, regions_split)])
    return merged


def adjacency_matrix_bipartite(data, key, column1, column1_split, column2, column2_split):
    G = nx.Graph()
    # If [1, 2, 3] is string type
    if isinstance(data[column1].iloc[0], str) and data[column1].str.startswith('[').iloc[0]:
        data[column1] = data[column1].apply(eval)
    if isinstance(data[column2].iloc[0], str):
        data[column2] = data[column2].apply(eval)    
    # Node for column1 and column2
    x_list = set(data[column1].apply(lambda x: pd.Series(x)).stack().unique()) # Because the values in column1 are already stored as a list, not as a string
    y_list = set(data[column2].apply(lambda x: pd.Series(x)).stack().unique())
    G.add_nodes_from(x_list, bipartite=0)
    G.add_nodes_from(y_list, bipartite=1)
    # Iterate over the dataframe and add edges
    for idx, row in data.iterrows():
        id_ = row[key]
        try:
            x_key = row[column1].split(column1_split)
        except: # when row[column1] is already splitted in a list
            x_key = row[column1]
        y_key = row[column2]
        for x in x_key:
            for y in y_key:
                G.add_edge(x, y, key=id_)
    # Convert the bipartite graph into a projected graph and then into an edge list dataframe
    # y_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    # projected_graph = nx.bipartite.projected_graph(G, y_nodes)
    # edge_list = pd.DataFrame(list(projected_graph.edges(data=True)), columns=[column1, column2, 'key'])
    df_edgeList = pd.DataFrame(G.edges(data=True), columns=[column1, column2, 'attributes'])
    df_edgeList['attributes'] = df_edgeList['attributes'].apply(lambda x: x['key'])
    # reorder the columns
    new_order = ["attributes",column1, column2]
    df_edgeList = df_edgeList.reindex(columns=new_order)
    df_edgeList['key_count'] = df_edgeList.groupby('attributes')['attributes'].transform('count')
    subset_df_edgeList = df_edgeList.drop_duplicates(subset='attributes', keep='first')
    filter_edgeList = subset_df_edgeList[[column1, column2, "key_count"]]
    # Remove rows with empty strings
    filter_edgeList = filter_edgeList.replace("", float("nan")).dropna(subset=[column1, column2], how="any")   
    # Create an empty adjacency matrix with the same length as the node list
    to_adjacency_matrix = pd.DataFrame(0, index=x_list, columns=y_list)
    # Iterate over the rows of the dataframe and update the adjacency matrix
    for row in filter_edgeList.itertuples(index=False):
        col1, col2, freq = row
        to_adjacency_matrix.loc[col1, col2] += freq
    return df_edgeList, to_adjacency_matrix


  
def frequency(data, region):
    if isinstance(data.iloc[0], str):
        data2 = data.apply(eval)
    if isinstance(data.iloc[0], list):
        data2 = data.copy()
    # Flatten the column in to a single list
    topics = data2.explode().tolist()
    frequencies = Counter(topics)
    # Convert series to dataframe
    freq_by_region = pd.DataFrame({"region":region*len(frequencies), 
                                'topic': list(frequencies.keys()),
                                'frequency': list(frequencies.values())})
    return freq_by_region