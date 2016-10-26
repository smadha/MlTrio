from simple_expansion import simple_expansion_feature as simp
from collections import Counter

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
    ## 1 -> num of question NOT ans
    ## 2 -> tags of question ans
    ## 3 -> tags of question NOT ans
    def get_feature_train(self):
        #TODO EXCLUDE TRAINING QUESTION BEFORE CALCULATING FEARURE
        feature = [len(self.question_ans),len(self.question_not_ans) ]
            
        ans_tags = Counter([])
        for ques in self.question_ans:
            ans_tags.update(simp.get_user_tag(simp.questions[ques]))
        
        feature.extend(simp.get_one_feature(ans_tags, simp.question_tags))
          
        not_ans_tags = Counter([])
        for ques in self.question_not_ans:
            not_ans_tags.update(simp.get_user_tag(simp.questions[ques]))
        
        feature.extend(simp.get_one_feature(not_ans_tags, simp.question_tags))
        
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
    ## 2 -> tags of user ans
    ## 3 -> tags of user NOT ans
    def get_feature_train(self):
        
        feature = [len(self.user_ans),len(self.user_not_ans)]
        
        ans_tags = Counter([])
        for user in self.user_ans:
            ans_tags.update(simp.get_user_tag(simp.users[user]))
        
        feature.extend(simp.get_one_feature(ans_tags, simp.user_tags))
         
        not_ans_tags = Counter([])
        for user in self.user_not_ans:
            not_ans_tags.update(simp.get_user_tag(simp.users[user]))
        
        feature.extend(simp.get_one_feature(not_ans_tags, simp.user_tags))
            
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
        
    
    return user_to_feature,ques_to_feature

user_to_feature,ques_to_feature = get_history_feature()

def get_consolidated_feature_train(question, user):
    '''
    Return history feature for given user, question, label. History is calculated excluding current label.
    '''
    # simplify objects to feature
    user_f = user_to_feature[user].get_feature_train()
    question_f = ques_to_feature[question].get_feature_train()
        
    return user_f + question_f

if __name__ == '__main__':
    
    print get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "7b4f71989c4cefb93a1c639940aa032e", '0')
    print get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "7b4f71989c4cefb93a1c639940aa032e", '1')
    
    
    