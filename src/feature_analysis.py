from collections import Counter

def print_user_answered():
    users_answered = Counter({})
    questions_answered = Counter({})
    with open("../bytecup2016data/invited_info_train_class1.txt") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            if label == '1':
                question = training_data[0]
                user = training_data[1]
                users_answered.update([user])
                questions_answered.update([question])
            
            training_data = f.readline().strip().split("\t")
            
#     for user in users_answered.most_common():
#         print user[0],  user[1]   
#     
    
    for question in questions_answered.most_common():
        print question[0],  question[1]   
            
        

if __name__ == '__main__':
    print_user_answered()