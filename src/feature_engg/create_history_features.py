from simple_expansion import simple_expansion_feature as simp
from collections import Counter
import numpy as np
import mltrio_utils

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
    ## 0 -> num of question_id ans
    ## 1 -> num of question_id NOT ans
    ## 2 -> num of time similar tags question_id ans
    ## 3 -> num of time unsimilar tags question_id ans
    ## 4 -> num of time similar tags question_id NOT ans
    ## 5 -> num of time unsimilar tags question_id NOT ans

    def get_feature_train(self, ques_asked):
        #TODO EXCLUDE TRAINING QUESTION BEFORE CALCULATING FEATURE
        
        # 0,1
        feature = [len(self.question_ans),len(self.question_not_ans) ]
        q_tag = simp.get_question_tag(simp.questions[ques_asked])[0]
        
        ans_tags = Counter([])
        for ques in self.question_ans:

            ans_tags.update(simp.get_question_tag(simp.questions[ques]))
          
        not_ans_tags = Counter([])
        for ques in self.question_not_ans:
            not_ans_tags.update(simp.get_question_tag(simp.questions[ques]))
        
        
        # 2,3
        feature.append(ans_tags[q_tag])
        ans_tags[q_tag] = 0
        feature.append(sum(ans_tags.values()))
        
        # 4,5
        feature.append(not_ans_tags[q_tag])
        not_ans_tags[q_tag] = 0
        feature.append(sum(not_ans_tags.values()))

        # 6,7
        # TODO: do the same for SD, max, min
        L1_dist = 0.0
        L2_dist = 0.0
        for ques in self.question_ans:
            L1_dist += mltrio_utils.get_L1_dist(simp.get_ques_feature(simp.questions[ques_asked]),
                                   simp.get_ques_feature(simp.questions[ques]))

        for ques in self.question_ans:
            L2_dist += mltrio_utils.get_L2_dist(simp.get_ques_feature(simp.questions[ques_asked]),
                                   simp.get_ques_feature((simp.questions[ques])))
        
        # if len(self.question_ans) = 0 give a higher distance 
        if (len(self.question_ans) == 0): 
            feature.append(1000)
            feature.append(1000)
        else:
            feature.append(L1_dist / len(self.question_ans))
            feature.append(L2_dist / len(self.question_ans))
        L1_dist = 0.0
        L2_dist = 0.0
        for ques in self.question_not_ans:
            L1_dist += mltrio_utils.get_L1_dist(simp.get_ques_feature(simp.questions[ques_asked]),
                                   simp.get_ques_feature(simp.questions[ques]))

        for ques in self.question_not_ans:
            L2_dist += mltrio_utils.get_L2_dist(simp.get_ques_feature(simp.questions[ques_asked]),
                                   simp.get_ques_feature(simp.questions[ques]))
        
        # if len(self.question_not_ans) = 0 give a higher distance
        if(len(self.question_not_ans) == 0):
            feature.append(1000)
            feature.append(1000)
        else: 
            feature.append(L1_dist / len(self.question_not_ans))
            feature.append(L2_dist / len(self.question_not_ans))
        
        return feature

class QuesHistory:
    'Class to store not history based features for all question_id'
    user_ans = [] ##IDs of users who've answered this question_id
    user_not_ans = [] ##
    def __init__(self):
        self.user_ans = []
        self.user_not_ans = []

    def add_user_ans(self, user_id):
        '''

        :param user_id: user_id ID
        :return:
        '''
        self.user_ans.append(user_id)
    
    def add_user_not_ans(self, user_id):
        '''

        :param user_id: user_id ID
        :return:
        '''
        self.user_not_ans.append(user_id)
        
    ## feature array format
    ## 0 -> num of user_id ans
    ## 1 -> num of user_id not ans
    ## 2 -> num of time similar tag of user_id ans
    ## 3 -> num of time unsimilar tag of user_id ans
    ## 4 -> num of time similar tag of user_id NOT ans
    ## 5 -> num of time unsimilar tag of user_id NOT ans

    def get_feature_train(self, user_target):
        '''

        :param user_target: user_id ID
        :return: all features based on this question_id and user_id ID
        '''
        feature = [len(self.user_ans),len(self.user_not_ans)]
        
        u_tag = simp.get_user_tag(simp.users[user_target])
        
        ans_tags = Counter([])
        for user_id in self.user_ans:
            ans_tags.update(simp.get_user_tag(simp.users[user_id]))
                 
        not_ans_tags = Counter([])
        for user_id in self.user_not_ans:
            not_ans_tags.update(simp.get_user_tag(simp.users[user_id]))
        
        
        # 2,3
        sim_tag_ans = 0
        for tag in u_tag:
            sim_tag_ans+=ans_tags[tag]
            ans_tags[tag] = 0
            
        feature.append(sim_tag_ans)
        feature.append(sum(ans_tags.values()))
        
        # 4,5
        not_sim_tag_ans = 0
        for tag in u_tag:
            not_sim_tag_ans+=not_ans_tags[tag]
            not_ans_tags[tag] = 0
        
        feature.append(not_sim_tag_ans)
        feature.append(sum(not_ans_tags.values()))    

        # 6,7
        # TODO: do the same for SD, max, min, handle zero error
        L1_dist = 0.0
        L2_dist = 0.0
        for user_id in self.user_ans:
            L1_dist += mltrio_utils.get_L1_dist(simp.get_user_feature(simp.users[user_target]),simp.get_user_feature(simp.users[user_id]))

        for user_id in self.user_ans:
            L2_dist += mltrio_utils.get_L2_dist(simp.get_user_feature(simp.users[user_target]),simp.get_user_feature((simp.users[user_id])))

        feature.append(L1_dist/len(self.user_ans))
        feature.append(L2_dist/len(self.user_ans))
        L1_dist = 0.0
        L2_dist = 0.0
        for user_id in self.user_not_ans:
            L1_dist += mltrio_utils.get_L1_dist(simp.get_user_feature(simp.users[user_target]), simp.get_user_feature(simp.users[user_id]))

        for user_id in self.user_not_ans:
            L2_dist += mltrio_utils.get_L2_dist(simp.get_user_feature(simp.users[user_target]), simp.get_user_feature(simp.users[user_id]))

        feature.append(L1_dist / len(self.user_not_ans))
        feature.append(L2_dist / len(self.user_not_ans))

        return feature


            
def get_history_feature():
    '''
    return dictionary<user_id,feature_array>
    We calculate feature for all users at once and return dictionary<user_id,feature_array> 
    which can be referenced multiple times
    '''
    
    ## Dictionary<user_id, UserHistory> to store feature array for all user_id   
    user_to_feature = {}
    ## Dictionary<ques_id, QuesHistory> to store feature array for all questions   
    ques_to_feature = {}
    
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            question_id = training_data[0]
            user_id = training_data[1]
            #fill dict if empty
            if user_id not in user_to_feature: user_to_feature[user_id] = UserHistory() 
            if question_id not in ques_to_feature: ques_to_feature[question_id] = QuesHistory()
            
            if label == '1' :
                user_to_feature[user_id].add_ques_ans(question_id)
                ques_to_feature[question_id].add_user_ans(user_id)
            if label == '0' :
                user_to_feature[user_id].add_ques_not_ans(question_id)
                ques_to_feature[question_id].add_user_not_ans(user_id)
                
            training_data = f.readline().strip().split("\t")
        
    
    return user_to_feature,ques_to_feature

user_to_feature,ques_to_feature = get_history_feature()

def get_consolidated_feature_train(question_id, user_id):
    '''
    Return history feature for given user_id, question_id, label. History is calculated excluding current label.
    '''
    # simplify objects to feature
    user_f = user_to_feature[user_id].get_feature_train(question_id)
    question_f = ques_to_feature[question_id].get_feature_train(user_id)
        
    return user_f + question_f

if __name__ == '__main__':
    
    print get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "7b4f71989c4cefb93a1c639940aa032e")
    print get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "7b4f71989c4cefb93a1c639940aa032e")
    qh = QuesHistory()
    qh.add_user_ans("7b4f71989c4cefb93a1c639940aa032e")

    print qh.get_feature_train("7b4f71989c4cefb93a1c639940aa032e")

    