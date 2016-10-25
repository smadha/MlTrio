
import cosine_similarity

def build_user_data():
    from  simple_expansion_feature import get_user_feature, users
    
    user_features_list = []
    for key in users.keys():
        #print smple_exp.users[key]
        user = users[key]
        user_features_list.append(get_user_feature(user))

    return user_features_list

user_features_list = build_user_data()
print user_features_list[0]

db = cosine_similarity.run_DBScan(user_features_list[0:10000])