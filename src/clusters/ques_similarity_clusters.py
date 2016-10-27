import dbscan_clustering
from  src.simple_expansion.simple_expansion_feature import get_ques_feature, questions
import cPickle as pickle

def build_ques_data():
    ques_features_list = []
    for key in questions.keys():
        #print smple_exp.users[key]
        question = questions[key]
        ques_features_list.append(get_ques_feature(question))

    return ques_features_list

def build_ques_cluster_dict(db):
    
    questions_cluster_dict = dict()
    cluster_questions_dict = dict()
    i = 0
    for key in questions.keys():
        cluster_id = db.labels_[i]
        questions_cluster_dict[key] = cluster_id
        if cluster_id in cluster_questions_dict.keys() and (not cluster_questions_dict[cluster_id] is None):
            cluster_questions_dict[cluster_id].append(key)
        else:
            cluster_questions_dict[cluster_id] = [key]
         
        i += 1   
        
    pickle.dump(cluster_questions_dict, open("cluster_ques_dict.p", "wb") )
    pickle.dump(questions_cluster_dict, open("ques_cluster_dict.p", "wb") )


ques_features_list = build_ques_data()
db = dbscan_clustering.run_DBScan(ques_features_list)
build_ques_cluster_dict(db)
