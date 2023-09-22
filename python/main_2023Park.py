from modules.GlobalVariables import *
from modules.NetworkAnalysis import *
from modules.preprocess import *
from modules.test import *

import datetime
import pytz

def main():
    # Print the current time in South Korea
    current_time_korea_started = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Started:", current_time_korea_started.strftime("%Y-%m-%d %H:%M"))

    run_ = {"readDataFolder": False, "run_filter_column": False, "run_unify_affiliation": False,
                  "run_descriptive_analysis": False, "run_network_analysis": True, "run_bipartite_network_analysis": False}
    doc_type_ = "patent"

    # result = test()

    # Load and merge data
    result = readData_folder(paper_ = False, patent_ = True, run=run_["readDataFolder"])
    
    # Filter out unnecessary columns    
    preprocess_filter = PreprocessFilter(doc_type=doc_type_) # paper OR patent
    result = preprocess_filter.filter_column(run=run_["run_filter_column"])

    # Unify the applications' names and match with regions    
    unify_names = UnifyNames(doc_type=doc_type_)
    result = unify_names.unify_affiliation(run=run_["run_unify_affiliation"])
    
    # run_match_region = False # match affiliations with region...for patents
    # result = unify_names.match_region(run=run_match_region)

    # Descriptive analysis    
    descriptive_analysis = Descriptive(doc_type=doc_type_)
    collab_ = {"collab": True, "overall": True, "only_selected": True} # annual trend of collabrated publications / annual trend of all publications
    result = descriptive_analysis.descriptive_(collab_, run=run_["run_descriptive_analysis"])

    # Network analysis    
    network_analysis = NetworkAnalysis(doc_type=doc_type_)
    network_ = {"collab": False, "knowledge": True} # collaboration / knowledge network
    result = network_analysis.network_analysis_byType(network_, run=run_["run_network_analysis"])
    # Bipartite analysis
    result = network_analysis.bipartite_network(run=run_["run_bipartite_network_analysis"])


    current_time_korea_finished = datetime.datetime.now(pytz.timezone("Asia/Seoul"))
    print("Ended:", current_time_korea_finished.strftime("%Y-%m-%d %H:%M"))

    return result
    


if __name__ == "__main__":
    main()