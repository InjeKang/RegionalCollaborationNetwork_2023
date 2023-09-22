from modules.GlobalVariables import *

import pandas as pd
import swifter
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import numbers
import itertools
import math


def strategic_diagram(strategicDiagram):
    # read patent data
    df1 = read_data("(3)US특허만 필요한 필드만.xlsx", "수소전지US 필요필드만")
    df2 = read_data("(3)US특허만 필요한 필드만.xlsx", "수전해US 필요필드만")
    # merge data
    data = pd.concat([df1, df2])
    data = data.reset_index(drop=True)
    # remove duplicated rows
    data2 = data.drop_duplicates(subset="WIPS ON key")
    data2 = data2.reset_index(drop=True)

    # select relevant columns
    data2 = data2[["WIPS ON key", "출원일", "Current IPC All", "출원인 국적"]]
    data2 = data2.rename(columns = {"WIPS ON key":"key", "Current IPC All":"ipc", "출원인 국적":"nationality"})
    # four-digit-IPC
    data2["ipc_7digit"] = data2.swifter.apply(lambda x: seven_digit(x["ipc"].split(" | ")), axis = 1)
    # select year and convert from str to numeric
    data2["year"] = data2.swifter.apply(lambda x: get_year(x["출원일"]), axis=1)
    data2["year"] = pd.to_numeric(data2["year"], errors="coerce")
    data2 = data2[["key", "year", "ipc_7digit", "nationality"]]

    subPeriod = [2014, 2015, "total"]
    nation_ = ["JP", "KR"]
    # subset_list = [(year, nation) for year, nation in itertools.product(subPeriod, nation_)]
    
    # for year, nation in subset_list:
    for sub_ in subPeriod:
        if sub_ == 2014:
            subPeriod_df = data2[data2["year"] <= sub_]
        elif sub_ == 2015:
            subPeriod_df = data2[data2["year"] >= sub_]
        else:
            subPeriod_df = data2.copy()        
        df = []
        for nation in nation_:            
            data3 = subPeriod_df[subPeriod_df["nationality"].str.contains(nation, case=False)]
            # read data with IPC matched with cluster
            cluster_data = read_data(f"04.merged_network&cluster_{sub_}.xlsx", nation)
            cluster_data = cluster_data.rename(columns = {"ipc": "ipc_7digit"})
            cluster_data = cluster_data[["ipc_7digit", "cluster"]]            
            # seperate the lists in the IPC column into individual rows and merge the exploded dataframe
            merged_data = pd.merge(data3.explode("ipc_7digit"), cluster_data, on="ipc_7digit")
            merged_data = merged_data[["key", "year", "ipc_7digit", "cluster"]]
            # drop rows with NaN
            merged_data.dropna(subset=["cluster"], inplace=True)
            merged_data["cluster"] = merged_data["cluster"].astype(int)

            # measure NGI
            merged_data2 = measure_ngi(merged_data)

            # measure NPI
            merged_data3 = measure_npi(merged_data2)
            
            # remove duplicate values
            merged_data3.drop_duplicates(subset=["cluster"], keep="first", inplace=True)

            # plot the data            
            plt.scatter(merged_data3["npi"], merged_data3["ngi"])            
            plt.xlabel("Normalized Performance Index")
            plt.ylabel("Normalized Growth Index")            
            plt.title(f"Strategic Diagram_{sub_}_{nation}")       
            plt.axhline(0, color="black", linestyle="-")
            plt.axvline(0, color="black", linestyle="-")
            merged_data3 = merged_data3.reset_index()
            for i in range(len(merged_data3)):
                plt.text(merged_data3["npi"][i], merged_data3["ngi"][i], merged_data3["cluster"][i], va='bottom', ha='center')             
            plt.savefig(f"results\\Strategic Diagram_{sub_}_{nation}.png")
            plt.clf()            

            # merge the results and save in excel
            df.append(merged_data3)            
        with pd.ExcelWriter(f"data\\05.StrategicDiagram_{sub_}.xlsx") as writer:
            df[0].to_excel(writer, sheet_name = "JP", index=False)
            df[1].to_excel(writer, sheet_name = "KR", index=False)



def measure_npi(data):
    data["tp"] = data.groupby("cluster")["cluster"].transform("count")
    data["log_tp"] = data.swifter.apply(lambda x: math.log(x["tp"]), axis = 1)
    tp_mean = data["log_tp"].mean()
    tp_sd = data["log_tp"].std()
    data["npi"] = data.swifter.apply(lambda x: np.nan if tp_sd == 0
                                       else (x["log_tp"]-tp_mean)/tp_sd, axis = 1)
    # tp_mean = data["tp"].mean()
    # tp_sd = data["tp"].std()
    # data["npi"] = data.swifter.apply(lambda x: np.nan if tp_sd == 0
    #                                    else (x["tp"]-tp_mean)/tp_sd, axis = 1)                                       
    return data


def measure_ngi(data):
    # measure GI of each cluster
    data["mean"] = data.groupby("cluster")["year"].transform("mean")
    data["min"] = data.groupby("cluster")["year"].transform("min")
    data["max"] = data.groupby("cluster")["year"].transform("max")
    data["gi"] = data.swifter.apply(lambda x: ((x["mean"] - x["min"]) / (x["max"] - x["min"])) 
                                        if x["max"] - x["min"] != 0 else np.nan, axis = 1)

    # measure NGI of every IPCs
    gi_mean = data["gi"].mean()
    gi_sd = data["gi"].std()
    data["ngi"] = data["gi"].swifter.apply(lambda x: (x-gi_mean)/gi_sd, axis = 1)
    data2 = data.sort_values(by="key")
    data2.reset_index()
    return data2



