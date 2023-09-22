from modules.GlobalVariables import *
from modules.NetworkAnalysis import *
from modules.StrategicDiagram import *
from modules.LDATopicModel import *
from modules.preprocess import *

def main():
    # Load and merge data
    readDataFolder = False
    result = readData_folder(paper_ = False, patent_ = True, run=readDataFolder)
    
    # Filter out unnecessary columns
    sampleData = False
    result = preprocess_data(paper_ = False, patent_ = True, run=sampleData)

    # LDA topic modeling
    corpusCleansing = False
    result = lda_optimal_no(raw_paper_ = False, raw_patent_ = False, 
                                   cleansed_paper_ = False, cleansed_patent_ = False, run=corpusCleansing)
    
    topicModel = True    
    opt_no = 4 # optimal number of topics extracted from the function 'lda_optimal_no'
    result = lda_topicModeling(opt_no, paper_ = False, patent_ = True, run=topicModel)
    
    # Knowledge network
    biPartite = True
    column_ = ["ipc_4digit", "topic_tags"]
    result = biPartite_network(opt_no, column_, paper_ = False, patent_ = True, run=biPartite)

    # Collaboration network
    collaboration_ = True
    sample_ = ["region", "applicant_region"]
    for i in sample_:
        result = collaboration_network(i, opt_no, paper_ = False, patent_ = True, run=collaboration_)
    return result
    
    



if __name__ == "__main__":
    main()