from simple_expansion import simple_expansion_feature as simp

class UserHistory:
    'Class to store history based features for all users'
    question_ans = []
    question_not_ans = []
    def __init__(self):
        self.question_ans = []
        self.question_not_ans = []
        
    def add_ques_ans(self, ques):
        self.question_ans.append(ques)
    
    def add_ques_not_ans(self, ques):
        self.question_not_ans.append(ques)
    
    ## feature array format
    ## 0 -> num of question ans
    ## 1 -> num of question not ans
    def get_feature(self):
        feature = [len(self.question_ans),len(self.question_not_ans)]
        return feature

class QuesHistory:
    'Class to store not history based features for all question'
    user_ans = []
    user_not_ans = []
    def __init__(self):
        self.user_ans = []
        self.user_not_ans = []

    def add_user_ans(self, user):
        self.user_ans.append(user)
    
    def add_user_not_ans(self, user):
        self.user_not_ans.append(user)
        
    ## feature array format
    ## 0 -> num of user ans
    ## 1 -> num of user not ans
    def get_feature(self):
        feature = [len(self.user_ans),len(self.user_not_ans)]
        return feature
            
def get_history_feature():
    '''
    return dictionary<user_id,feature_array>
    We calculate feature for all users at once and return dictionary<user_id,feature_array> 
    which can be referenced multiple times
    '''
    
    ## Dictionary<user_id, UserHistory> to store feature array for all user   
    user_to_feature = {}
    ## Dictionary<ques_id, QuesHistory> to store feature array for all questions   
    ques_to_feature = {}
    
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            question = training_data[0]
            user = training_data[1]
            #fill dict if empty
            if user not in user_to_feature: user_to_feature[user] = UserHistory() 
            if question not in ques_to_feature: ques_to_feature[question] = QuesHistory()
            
            if label == '1' :
                user_to_feature[user].add_ques_ans(question)
                ques_to_feature[question].add_user_ans(user)
            if label == '0' :
                user_to_feature[user].add_ques_not_ans(question)
                ques_to_feature[question].add_user_not_ans(user)
                
            training_data = f.readline().strip().split("\t")
        
    
    # simplify objects to feature
    for key in user_to_feature:
        user_to_feature[key] = user_to_feature[key].get_feature()
    
    for key in ques_to_feature:
        ques_to_feature[key] = ques_to_feature[key].get_feature()
        
    return user_to_feature,ques_to_feature

if __name__ == '__main__':
    print get_history_feature()