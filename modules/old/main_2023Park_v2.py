from modules.GlobalVariables import *
from modules.NetworkAnalysis import *
from modules.StrategicDiagram import *
from modules.LDATopicModel import *

def main():
    # read data
    readDataFolder = False
    result = readData_folder(paper_ = False, patent_ = True, run=readDataFolder)
    
    main_func = functions_(
        # read data
        readDataFolder = False
        , sampleData = False
        # LDA-topic model
        , ldaTopic = True
        # network analysis
        , knowledgeNetwork = False
        , cluster_ = False
        , strategicDiagram = False
    )

def functions_(readDataFolder, sampleData, ldaTopic, knowledgeNetwork, cluster_, strategicDiagram):
    
    
    if readDataFolder:
        result = readData_folder(paper_ = False, patent_ = True)    
    if sampleData:
        result = preprocess_data(paper_ = False, patent_ = True)
    if ldaTopic:
        data_cleansing_process = False
        # to run corpus cleansing process
        if data_cleansing_process == True:
            data_cleansed = lda_optimal_no(raw_paper_ = False, raw_patent_ = False,
                                            cleansed_paper_ = False, cleansed_patent_ = False)
        else:
            type_ = "patent" # patent OR paper
            data_cleansed = pd.read_pickle(f"data\\03.{type_}_LDA_data_cleansed.pkl")
        # if cleansing process has been already conducted
        opt_no = 6 # optimal number of topics extracted from the function 'lda_optimal_no'
        result = lda_topicModeling(data_cleansed, opt_no, paper_ = False, patent_ = True)

    if knowledgeNetwork:
        result = knowledge_network("app_reg", True, False) # region/app_reg/ipc_7digit, destriptive, toEdgeList
    if cluster_: 
        result = cluster_network(cluster_)
    if strategicDiagram:
        result = strategic_diagram(strategicDiagram)
    try:
        return result
    except:
        return data_cleansed


if __name__ == "__main__":
    main()