import graphlab as gl


user_arr = []
item_arr = []

INVITED_INFO_TRAIN = "../../bytecup2016data/invited_info_train_PROC.txt"

with open(INVITED_INFO_TRAIN) as f:
    training_data = f.readline().strip().split("\t")
    while training_data and len(training_data) == 3 :
        
        label = training_data[2]
        
        question = training_data[0]
        user = training_data[1]
        
        if label == "1":  
            user_arr.append(user)
            item_arr.append(question)
        
        training_data = f.readline().strip().split("\t")
        
print len(user_arr)

sf = gl.SFrame({'user_id':  user_arr, 'item_id': item_arr })

# train, test = gl.recommender.util.random_split_by_user(sf)

train = sf

# |   1    | 0.0386473429952 | 0.0236178207193 |  0.001
# |   1    | 0.0531400966184 | 0.0271604938272 |  0.1
# |   1    | 0.0458937198068 | 0.0259891879457 |  1.0


m3 = gl.ranking_factorization_recommender.create(train, ranking_regularization = 1, unobserved_rating_value = 0.4, verbose=False)

m3 = gl.ranking_factorization_recommender.create(train, ranking_regularization = 1, unobserved_rating_value = 0.4, verbose=False)

user_arr_t = []
item_arr_t = []
label_arr_t = []

with open("../../bytecup2016data/validate_nolabel.txt","r") as f:
    training_data = f.readline().strip().split(",")
    training_data = f.readline().strip().split(",")
    print training_data
    while training_data and len(training_data) == 2 :
        question = training_data[0]
        user = training_data[1]
        
        user_arr_t.append(user)
        item_arr_t.append(question)
    
        training_data = f.readline().strip().split(",")

test = gl.SFrame({'user_id':  user_arr_t, 'item_id': item_arr_t })      
label_arr_t = m3.predict(test)

with open("../../bytecup2016data/validate_label_graphlab.csv", "w") as f:
    f.write("qid,uid,label\n")
    for user, item, label in zip(user_arr_t, item_arr_t, label_arr_t):
        f.write(item + "," + user + "," + str(label) + "\n" )

# report = m3.evaluate_precision_recall(test, exclude_known = True, verbose=False)

# report = m3.evaluate(test, exclude_known_for_precision_recall=True, verbose=False)

# print report["precision_recall_overall"]
