from modules.GlobalVariables import *

import pandas as pd
import swifter
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



    

def knowledge_network(column_, descriptive, toEdgeList):
    data = read_data("00.patent_merged_v2(cleansed).xlsx", "Sheet1")
    data2 = data.copy()    
    data2["year"] = data2["year"].replace([np.inf, -np.inf], np.nan)
    data2 = data2.dropna() # 78494 >> 76308
    data2["year"] = data2["year"].astype(int)  # Convert year column to integers
    data2['app_reg'] = data2.apply(lambda row: ' | '.join([f'{app}({reg})' for app, reg in zip(row['applicants'].split(' | '), row['region'].split(' | '))]), axis=1)
    region_list = ["경상북도", "경상남도", "부산", "대구", "울산"]

    if descriptive:        
        df_merged_overall = []
        df_merged_collab = []
        for region in region_list:            
            sub_data = data2[data2["region"].str.contains(region, na=False)]

            # generation trend
            # Group by year and count the number of patents
            patents_by_year = sub_data.groupby("year").size()
            df_merged_overall.append(patents_by_year)
            # Plot the number of patents by year
            patents_by_year.plot(kind="line", marker="o")
            plt.xlabel("Year")
            plt.ylabel("Number of Patents")
            plt.title(f"Number of Patents in {region} by Year")
            plt.xticks(range(min(sub_data["year"]), max(sub_data["year"]) + 1))
            plt.savefig(f"results\\Annual_Trend_overall_{region}.png")
            plt.clf()

        df_merged_overall2 = pd.concat(df_merged_overall, axis=1, keys=region_list)
        df_merged_overall2 = df_merged_overall2.round(1)
        average_row = df_merged_overall2.mean(axis=0).to_frame().T
        df_merged_overall2 = df_merged_overall2.append(average_row, ignore_index=False)
        df_merged_overall2 = df_merged_overall2.rename(index={df_merged_overall2.index[-1]: "avg"})

        # save as an excel file
        with pd.ExcelWriter(f"results\\01.descriptive_trend_overall_{column_}.xlsx") as writer:
            df_merged_overall2.to_excel(writer, sheet_name = "overall")
            for i in range(len(df_merged_overall)):
                df_merged_overall[i].to_excel(writer, sheet_name = region_list[i], index=False)

            # collaboration trend
            # Create the "collab" column based on whether "|" is included in "region"
            sub_data["collab"] = sub_data["region"].apply(lambda x: "공동" if "|" in x else "단독")
            # Group by year and calculate the ratio of collaboration
            collab_ratio_by_year = sub_data[sub_data["collab"] == "공동"].groupby("year").size() / sub_data.groupby("year").size() * 100
            df_merged_collab.append(collab_ratio_by_year)
            # Plot the ratio of collaboration using a line plot
            collab_ratio_by_year.plot(kind="line", marker="o")
            plt.xlabel("Year")
            plt.ylabel("Percentage")
            plt.title(f"Ratio of Collaboration in {region} by Year")
            plt.xticks(range(min(sub_data["year"]), max(sub_data["year"]) + 1))
            # plt.grid(True)
            plt.savefig(f"results\\Annual_Trend_collab_ratio_{region}.png")
            plt.clf()            

        df_merged_collab2 = pd.concat(df_merged_collab, axis=1, keys=region_list)
        df_merged_collab2 = df_merged_collab2.round(1)
        average_row = df_merged_collab2.mean(axis=0).to_frame().T
        df_merged_collab2 = df_merged_collab2.append(average_row, ignore_index=False)
        df_merged_collab2 = df_merged_collab2.rename(index={df_merged_collab2.index[-1]: "avg"})

        with pd.ExcelWriter(f"results\\01.descriptive_trend_collab_{column_}.xlsx") as writer:
            df_merged_collab2.to_excel(writer, sheet_name = "overall")
            for i in range(len(df_merged_collab)):
                df_merged_collab[i].to_excel(writer, sheet_name = region_list[i], index=False)

    if toEdgeList:
        # for region in region_list:
        #     sub_data = data2[data2[column_].str.contains(region, na=False)]            
        #     EdgeList_df = edge_list(sub_data, column_)
        
        if column_ == "region":
            EdgeList_df = edge_list(data2, column_)
            EdgeList_sub = EdgeList_df[EdgeList_df['column1'].isin(region_list) & EdgeList_df['column2'].isin(region_list)]
            network_analysis(EdgeList_sub, column_, 0)
        if column_ == "app_reg":
            EdgeList_df = edge_list(data2, column_)
            EdgeList_sub = EdgeList_df[EdgeList_df['column1'].apply(lambda x: any(region in x for region in region_list))
                                       & EdgeList_df['column2'].apply(lambda x: any(region in x for region in region_list))]        
            network_analysis(EdgeList_sub, column_, 0)
        
        if column_ == "ipc_7digit":
            for region in region_list:                
                sub_data = data2[data2["region"].str.contains(region, na=False)] 
                EdgeList_df = edge_list(sub_data, column_)
                network_analysis(EdgeList_df, column_, region)
        # sub_data = data2[data2[column_].str.contains(region, na=False)]            
        # EdgeList_df = edge_list(sub_data, column_)
        # EdgeList_sub = EdgeList_df[EdgeList_df['column1'].isin(region_list) & EdgeList_df['column2'].isin(region_list)]

def network_analysis(data, column_, region):
    EdgeList_sub = data.copy()
    # measure network characteristics
    ipc_freq = {(a, b, len(g)) for (a, b), g in EdgeList_sub.groupby(['column1', 'column2'])}
    # Convert set to a list
    ipc_freq_list = list(ipc_freq)
    # Create dataframe
    edges = pd.DataFrame(ipc_freq_list, columns=["column1", "column2", "weight"])
    # Create a copy of the DataFrame with the columns sorted to handle both directions
    edges_sorted = edges.copy()
    edges_sorted[['column1', 'column2']] = np.sort(edges[['column1', 'column2']], axis=1)

    # Group the DataFrame by the sorted columns and sum the weights
    edges_combined = edges_sorted.groupby(['column1', 'column2'])['weight'].sum().reset_index()
    if column_ == "region":
        edges_combined.sort_values('weight', ascending=False).to_excel(f"data\\01.EdgeList_{column_}_overall.xlsx", index=False)
    if "app_reg":
        edges_combined = edges_combined.sort_values('weight', ascending=False)
        edges_combined = edges_combined[edges_combined["weight"]>=5]        
        edges_combined.sort_values('weight', ascending=False).to_excel(f"data\\01.EdgeList_{column_}_overall.xlsx", index=False)
    if "ipc_7digit":
        edges_list = edges_combined.values.tolist()
        edges_list2 = [(a, b, c) for [a, b, c] in edges_list]
        edges_combined.sort_values('weight', ascending=False).to_excel(f"data\\01.EdgeList_{column_}_{region}.xlsx", index=False)

    nodes = pd.unique(edges_combined[["column1", "column2"]].values.ravel("K"))
    
    # Create an empty adjacency matrix with the same length as the node list
    adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
    # Iterate over the rows of the dataframe and update the adjacency matrix
    if column_ == "region":
        for row in EdgeList_sub.itertuples(index=False):
            column1, column2 = row
            if column1 == column2:
                adjacency_matrix.loc[column1, column2] += 0.5
                adjacency_matrix.loc[column2, column1] += 0.5
            else:
                adjacency_matrix.loc[column1, column2] += 1
                adjacency_matrix.loc[column2, column1] += 1        
            # need to remove the column manually to use VosViewer
        adjacency_matrix.to_csv(f"data\\01.adjMatrix_{column_}_overall_v3.csv", encoding="utf-8-sig") 
    if column_ == "app_reg":
        edges_combined_iter = edges_combined[["column1", "column2"]]
        for row in edges_combined_iter.itertuples(index=False):
            column1, column2 = row
            if column1 == column2:
                adjacency_matrix.loc[column1, column2] += 0.5
                adjacency_matrix.loc[column2, column1] += 0.5
            else:
                adjacency_matrix.loc[column1, column2] += 1
                adjacency_matrix.loc[column2, column1] += 1        
            # need to remove the column manually to use VosViewer
        adjacency_matrix.to_csv(f"data\\01.adjMatrix_{column_}_overall.csv", encoding="utf-8-sig") 
    else:
        edges_combined_iter = edges_combined[["column1", "column2"]]
        for row in edges_combined_iter.itertuples(index=False):
            column1, column2 = row
            if column1 == column2:
                adjacency_matrix.loc[column1, column2] += 0.5
                adjacency_matrix.loc[column2, column1] += 0.5
            else:
                adjacency_matrix.loc[column1, column2] += 1
                adjacency_matrix.loc[column2, column1] += 1              
        adjacency_matrix.to_csv(f"data\\01.adjMatrix_{column_}_{region}.csv", encoding="utf-8-sig") 

    # # adjacency matrix
    # adj_matrix = pd.pivot_table(edges_combined, values='weight', index='column1', columns='column2', fill_value=0)
    # adj_matrix.to_csv(f"data\\01.adjMatrix_{column_}_overall_v2.csv", encoding="utf-8-sig") 

    nodes_df = edges["column1"].append(edges["column2"]).value_counts().reset_index()
    nodes_df.columns = ["column", "freq"]

    # identifying network structure
    G = nx.Graph()
    G.add_weighted_edges_from(edges_list2)
    # Draw the graph
    pos = nx.spring_layout(G)  # Position nodes using a spring layout
    edge_widths = [d['weight'] * 1/200 for (u, v, d) in G.edges(data=True)]  # Get edge weights
    # Plot the network graph
    # fig, ax = plt.subplots(figsize=(800, 600), dpi=100) # Set the figure size
    if column_ == "region" or column_ == "app_reg":
        nx.draw_networkx(G, pos, width=edge_widths, with_labels=True, font_family = font_name, node_color='lightblue', edge_color='gray')
        # Add edge labels (weights) to the figure
        # edge_labels = nx.get_edge_attributes(G, 'weight')
        # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        # plt.show()
        # plt.savefig(f"results\\NetworkResult_{region}.png")
        plt.savefig(f"results\\NetworkResult_{column_}_overall2.png")
        plt.clf()
    else:
        nx.draw_networkx(G, pos, width=edge_widths, with_labels=True, font_family = font_name, node_color='lightblue', edge_color='gray')
        plt.savefig(f"results\\NetworkResult_{column_}_{region}.png")
        plt.clf()
    # measuring three types of centrality        
    degree_centrality_ = nx.degree_centrality(G)
    closeness_centrality_ = nx.closeness_centrality(G)
    betweenness_centrality_ = nx.betweenness_centrality(G)
    constraints_ = nx.constraint(G)            
    clustering_coefficient_ = nx.clustering(G)
    # convert the results into a dataframe
    toDF = pd.DataFrame({
        "degree": pd.Series(degree_centrality_),
        "closeness": pd.Series(closeness_centrality_),
        "betweenness": pd.Series(betweenness_centrality_),
        "constraints": pd.Series(constraints_),
        "clustering" : pd.Series(clustering_coefficient_)
    })
    # from index to a column
    toDF.reset_index(inplace=True)
    toDF = toDF.rename(columns = {"index" : "column"})
    toDF = toDF.assign(structural_holes = 2 - toDF["constraints"])
    # merge two dataframes - toDF and nodes_df
    mergedDF = pd.merge(toDF, nodes_df, on="column", how = "outer")
    mergedDF = mergedDF[["column", "freq", "degree", "closeness", "betweenness", "structural_holes", "clustering", "constraints"]]
    mergedDF = mergedDF.sort_values(by="freq", ascending=False)
    # mergedDF.to_excel(f"data\\02.NetworkResults_{column_}_{region}.xlsx", index=False)
    if column_ == "region" or column_ == "app_reg":
        mergedDF.to_excel(f"data\\02.NetworkResults_{column_}_overall.xlsx", index=False)
    else:
        mergedDF.to_excel(f"data\\02.NetworkResults_{column_}_{region}.xlsx", index=False)

        
def cluster_network(cluster_):
    if cluster_:
        subPeriod = [2014, 2015, "total"]
        for sub_ in subPeriod:   
            nation_ = ["JP", "KR"]
            df = []
            for nation in nation_:
                network_char = read_data(f"02.NetworkResults_{nation}_{sub_}.xlsx", "Sheet1")
                cluster_result = read_data(f"03.ClusterResults_{sub_}.xlsx", nation)
                cluster_result = cluster_result.rename(columns = {"id":"ipc"})
                cluster_result = cluster_result[["ipc", "cluster"]]
                mergedDF = pd.merge(network_char, cluster_result, on="ipc", how = "outer")
                df.append(mergedDF)        
            with pd.ExcelWriter(f"data\\04.merged_network&cluster_{sub_}.xlsx") as writer:
                df[0].to_excel(writer, sheet_name = "JP", index=False)
                df[1].to_excel(writer, sheet_name = "KR", index=False)
