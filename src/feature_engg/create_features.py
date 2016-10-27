from simple_expansion import simple_expansion_feature as simp
from collections import Counter
import numpy as np
import mltrio_utils


def get_ques_user_similarity(user_id,ques_id):
    '''
    :param user_id: id of the user
    :param ques_id: id of the question
    :return: L1 and L2 distance between user and question data
    '''
    #TODO: remove the interest tags while comparing
    L1_dist = 0.0
    L2_dist = 0.0

    L1_dist += mltrio_utils.get_L1_dist(simp.get_ques_feature(simp.questions[ques_id]),
                                            simp.get_user_feature(simp.users[user_id]))
    L2_dist += mltrio_utils.get_L2_dist(simp.get_ques_feature(simp.questions[ques_id]),
                                            simp.get_user_feature((simp.users[user_id])))

    return L1_dist,L2_dist