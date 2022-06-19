#%%
from DataClean import Data
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from util import Data_twist


class Baseline():

    def __init__(self, data) -> None:
        self.df = Data_twist(data)
        self.prediction = [-1]*self.df.shape[0]
        # self.index_list = self.build_case_index()

    def Data_twist(self):
        '''
        Remove all the conflict data in the AAA tagging.
        '''
        self.df = self.df.fillna(0)
        for i in range(self.df.shape[0]):
            if self.df.loc[i,'AAA'] == '1,2':
                self.df.loc[i,'AAA'] = 2
            elif self.df.loc[i,'AAA'] == '0,1':
                self.df.loc[i,'AAA'] = 0
            elif self.df.loc[i,'AAA'] == '0,2':
                self.df.loc[i,'AAA'] = 0
            elif self.df.loc[i,'AAA'] == '0,1,2':
                self.df.loc[i,'AAA'] = 0
            elif self.df.loc[i,'AAA'] =='/':
                self.df.loc[i,'AAA'] = 0
            elif self.df.loc[i,'AAA'] == np.nan:
                self.df.loc[i,'AAA'] = 0
        return self.df
    
    def build_case_index(self):
        '''
        NOT USE RIGHT NOW.
        The index list contains start index and end index of each case.
        index_list = [start_index, end_index, start_index, end_index, ...]
        '''
        index_list = []
        method_name = ''
        for i in range(self.df.shape[0]):
            if method_name == '':
                method_name = self.df['testMethodName'][i]
                index_list.append(0)
            if method_name != self.df['testMethodName'][i]:
                index_list.append(i-1)
                index_list.append(i)
                method_name = self.df['testMethodName'][i]
            
        index_list.append(self.df.shape[0]-1)
        # print(index_list)
        # print(len(index_list))
        return index_list

    def remove_null(self, strlist):
        result_list = []
        for _ in strlist:
            if _ != '':
                result_list.append(_)
        return result_list

    def split_with_upper_case(self, method):
        #TODO: 定义了 大写连写 鉴定拼凑函数，比如 URL

        if type(method) == str:
            result_list = re.sub( r"([A-Z]+)", r" \1", method).split()
            return result_list
            
        if type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + re.sub( r"([A-Z]+)", r" \1", _).split()
            return self.remove_null(result_list)
    
    def remove_all_punctuantion(self, method):
        if type(method) == str:
            return method.split('_')
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + _.split('_')
            return self.remove_null(result_list)

    def IT_split(self,method):
        if type(method) == str:
            result_list = re.sub( r"(IT)", r" \1 ", method)
            result_list = re.sub(r"(VM)", r" \1 ", result_list).split()
            return result_list
        if type(method) == list:
            result_list = list()
            for _ in method:
                item = re.sub( r"(IT)", r" \1 ", _)
                item = re.sub(r"(VM)", r" \1 ", item)
                result_list = result_list + item.split()
            return self.remove_null(result_list)

    def lower_case(self, method):
        if type(method) == str:
            return method.lower()
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list.append(_.lower())
            return self.remove_null(result_list)

    def specific_twist(self, method):
        if type(method) == str:
            result_list = re.sub( r"(\d+)", r" \1 ", method).split()
            return result_list
        if type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + re.sub( r"(\d+)", r" \1 ", _).split()
            return self.remove_null(result_list)

    def name_match(self,statement_name, method_name):
        '''
        Name Match
        '''
        score = 0
        statement_name_processed = self.specific_twist(self.lower_case(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(statement_name.split('.')[-1])))))
        method_name_processed = self.specific_twist(self.lower_case(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(method_name)))))

        for word in statement_name_processed:
            if word in method_name_processed: # 可能出现多个相同的单词这样把相似词数推高
                score+=1

        return score

    
    def name_match_distribution(self):
        '''
        Using the Action name score distribution to calculate the threshold of the 
        Action name score. 
        '''
        action_name_score_list = []
        for i in range(self.df.shape[0]):
            if self.df['AAA'][i] == 1:
                action_name_score_list.append(self.name_match(self.df['potentialTargetQualifiedName'][i], self.df['testMethodName'][i]))
                if self.name_match(self.df['potentialTargetQualifiedName'][i], self.df['testMethodName'][i]) == 0:
                    print(self.df['potentialTargetQualifiedName'][i])
                    print(self.df['testMethodName'][i])
        unique, counts = np.unique(action_name_score_list, return_counts=True)

        print('The distribution of the Action name score is:')
        print(np.asarray((unique, counts)).T)

    def tag_filter(self):
        '''
        Filter the data with the tag.
        '''
        for i in range(self.df.shape[0]):
            try:
                keyword = self.df['potentialTargetQualifiedName'][i].lstrip().split(' ')[0]
                statement_name = self.df['potentialTargetQualifiedName'][i].split('.')[-1]
            except Exception as e:
                print(e)
                print('{0} line has no 0 position'.format(i))
            if keyword == 'ASSERT' or re.search(r'verify', statement_name, re.I) or re.search(r'check', statement_name, re.I):
                self.prediction[i] = 2
            elif keyword == 'NEW':
                self.prediction[i] = 0
    
    def n_biggest_in_dict(self, n, dict):
        '''
        Return the biggest n in dict.
        '''
        return sorted(dict.items(), key=lambda x: x[1], reverse=True)[:n]

    def case_rank(self, action_limits_in_one_case=2, assert_distance_limits=3):
        '''
        Rank the statments based on the name match and assert distance
        
        Limitation: There are multiple ways to calculate the name match,
        do you use the class name? do you use the edit distance to calculate the similarity?
        '''

        for i in range(self.df.shape[0]):
            if self.prediction[i] == -1:
        ###First: Assert Distance filter the arrangment ###
                if self.df['Assert Distance'][i] > assert_distance_limits:
                    self.prediction[i] = 0
                if self.df['Assert Distance'][i] == 0:
                    self.prediction[i] = 2

        ###Second: Name Match filter the action ###
                else:
                    score_dict = dict()
                    
                    for index in range(np.int32(self.df['Assert Distance'][i])):
                    # for index in range(assert_distance_limits):
                        try:
                            score_dict[str(index)] = self.name_match(self.df['potentialTargetQualifiedName'][i+index], self.df['testMethodName'][i+index])
                        except Exception as e:
                            print(e)
                            print(f'index is {index}')
                            print(f'i is {i}')
                    action_list = self.n_biggest_in_dict(action_limits_in_one_case, score_dict)
                    if self.prediction[i+index] == 2: #TODO:
                        print('ERROR: Assert is rewrited as Arrangment')
                        print(i+index)
                    self.prediction[i+index] = 0
                    for action in action_list:
                        self.prediction[i+int(action[0])] = 1
        
    def case_rank_with_edit_distance_or_Jcard(self, distribution_show=False):
        '''
        Replace the name match with the edit distance or Jcard number from the 
        data_edit or data_jacard.
        '''
        index_list = self.build_case_index()
        
        ### First: Name Similarity Normalization ###
        name_similarity_list = []
        for i in range(int(len(index_list)/2)):
            i = i*2
            # i is the start index, i+1 is the end index
            
            try:
                if index_list[i] == index_list[i+1]:
                    # hadle the case that only have one statement
                    case_name_similarity = np.array(self.df['Name Similarity'][index_list[i]])
                else:
                    case_name_similarity = self.df['Name Similarity'][index_list[i]:index_list[i+1]+1].values
                scaler = MinMaxScaler()
                # print(f"case similarity size before :{len(case_name_similarity)}")
                case_name_similarity = scaler.fit_transform(case_name_similarity.reshape(-1,1))
                # print(f"case similarity size after:{len(case_name_similarity)}")
            except Exception as e:
                print(e)
                print(f'index_list[i] is {index_list[i]}')
                print(f'index_list[i+1] is {index_list[i+1]}')
                print(f"case_name_similarity is {case_name_similarity}")
            for j in case_name_similarity:
                name_similarity_list.append(j[0])
        ### Second: Assert Distance Normalization ###
        assert_distance_list = []
        for i in range(int(len(index_list)/2)):
            i = i*2
            # i is the start index, i+1 is the end index
            if index_list[i] == index_list[i+1]:
                # hadle the case that only have one statement
                case_assert_distance = np.array(self.df['Assert Distance'][index_list[i]])
            else:
                case_assert_distance = self.df['Assert Distance'][index_list[i]:index_list[i+1]+1].values
            scaler = MinMaxScaler()
            case_assert_distance = scaler.fit_transform(case_assert_distance.reshape(-1,1))
            for j in case_assert_distance:
                assert_distance_list.append(j[0])
        ### Third: Action Relative Score Distritubtion###
        
        action_relative_score_list = []
        for _ in range(len(assert_distance_list)):
            action_relative_score = name_similarity_list[_] - assert_distance_list[_]
            action_relative_score_list.append(action_relative_score)

        # print(len(name_similarity_list))
        # print(len(assert_distance_list))
        # print(len(action_relative_score_list))
        # print(self.df.shape[0])
        if distribution_show == True:
            all_action_action_relative_score_list = []
            all_AR_AS_action_relative_score_list = []
            for i in range(self.df.shape[0]):
                if self.df["AAA"][i] == 1:
                    all_action_action_relative_score_list.append(action_relative_score_list[i])
                else:
                    all_AR_AS_action_relative_score_list.append(action_relative_score_list[i])
            return all_action_action_relative_score_list, action_relative_score_list, name_similarity_list, assert_distance_list, all_AR_AS_action_relative_score_list


        ### Calculate the distribution of the name similarity and assert distance in action, to find
        ### the threshold of the name similarity and assert distance.
        ### base on the distribution the threshold is <0.
        THREASHOLD = 0

         ### Thrashold filter ###
        for index in range(len(action_relative_score_list)):
            if action_relative_score_list[index] < THREASHOLD:
                self.prediction[index] = 1

        ### Rank Filter ###

        for i in range(int(len(index_list)/2)):
            i = i*2
            need_rank = True
            if index_list[i] == index_list[i+1]:
                # hadle the case that only have one statement
                case_statements = action_relative_score_list[index_list[i]:index_list[i+1]+1]
            else:
                case_statements = action_relative_score_list[index_list[i]:index_list[i+1]+1]

            for value in self.prediction[index_list[i]:index_list[i+1]+1]:
                if value == 0:
                    self.prediction[index_list[i]] = 1
                    need_rank = False
               
            if need_rank == True:
                for j in range(len(case_statements)):
                    if case_statements[j] == max(case_statements):
                        self.prediction[index_list[i]+j] = 1

    def remove_all_neg_one(self):
        for i in range(len(self.prediction)):
            if self.prediction[i] == -1:
                self.prediction[i] = 0

    def model(self,action_limits_in_one_case=2, assert_distance_limits=3):
        '''
        Tag -> Assert Distance -> Name Match by this way we can 
        avoid using consider the in case ranking, we only ranking the statents close to the assert.
        Ranking statements has assert distance 1,2,3.
        '''
        self.tag_filter()
        # self.case_rank(action_limits_in_one_case, assert_distance_limits)
        self.case_rank_with_edit_distance_or_Jcard()
        self.remove_all_neg_one()

        return self.prediction


def acc_kf(prob, y_test):
    count = 0
    target_count = 0
    target_acc_count = 0
    assert_count = 0
    assert_acc_count = 0
    for _ in range(len(prob)):
        if prob[_] == y_test[_]:
            count += 1
        if prob[_] == y_test[_] and prob[_] == 1:
            target_acc_count += 1
        if y_test[_] == 1:
            target_count +=1
        if prob[_] == y_test[_] and prob[_] == 2:
            assert_acc_count +=1
        if y_test[_] == 2:
            assert_count +=1
    acc = count/len(y_test)
    try:
        action_acc = target_acc_count/target_count
        print("Action Accuracy: {num}".format(num=action_acc))
    except Exception as e:
        print(e)
    # print(assert_count)
    # print(assert_acc_count)
    try:
        assert_acc = assert_acc_count/assert_count
        print("Assert accuracy: {num}".format(num=assert_acc))
    except Exception as e:
        print(e)
    print("accuracy : {num}".format(num=acc)) 
    
def recall_kf(predict, target):
    
    arrange_TP = 0
    arrange_FN = 0
    action_TP = 0
    action_FN = 0
    assert_TP = 0
    assert_FN = 0

    for _ in range(len(predict)):

        if predict[_] == target[_] and predict[_] == 0:
            arrange_TP += 1
        if predict[_] != target[_] and target[_] != 0:
            arrange_FN +=1
        if predict[_] == target[_] and predict[_] == 1:
            action_TP += 1
        if predict[_] != target[_] and target[_] != 1:
            action_FN +=1
        if predict[_] == target[_] and predict[_] == 2:
            assert_TP += 1
        if predict[_] != target[_] and target[_] != 2:
            assert_FN +=1

    try:
        arrange_recall = arrange_TP/(arrange_TP+arrange_FN)
        # print("Arrange Recall: {num}".format(num=arrange_recall))
    except Exception as e:
        print(e)
    # print(assert_count)
    # print(assert_acc_count)
    try:
        action_recall = action_TP/(action_TP+action_FN)
        # print("action recall: {num}".format(num=action_recall))
    except Exception as e:
        print(e)
    try:
        assert_recall = assert_TP/(assert_TP+assert_FN)
        # print("Assert recall: {num}".format(num=assert_recall))
    except Exception as e:
        print(e)
    return (arrange_recall,action_recall,assert_recall)

def percision_kf(predict, target):
    
    arrange_TP = 0
    arrange_FP = 0
    action_TP = 0
    action_FP = 0
    assert_TP = 0
    assert_FP = 0

    for _ in range(len(predict)):

        if predict[_] == target[_] and predict[_] == 0:
            arrange_TP += 1
        if predict[_] != target[_] and target[_] == 0:
            arrange_FP +=1
        if predict[_] == target[_] and predict[_] == 1:
            action_TP += 1
        if predict[_] != target[_] and target[_] == 1:
            action_FP +=1
        if predict[_] == target[_] and predict[_] == 2:
            assert_TP += 1
        if predict[_] != target[_] and target[_] == 2:
            assert_FP +=1

    try:
        arrange_per = arrange_TP/(arrange_TP+arrange_FP)
        # print("Arrange per: {num}".format(num=arrange_per))
    except Exception as e:
        print(e)
    # print(assert_count)
    # print(assert_acc_count)
    try:
        action_per = action_TP/(action_TP+action_FP)
        # print("action per: {num}".format(num=action_per))
    except Exception as e:
        print(e)
    try:
        assert_per = assert_TP/(assert_TP+assert_FP)
        # print("Assert per: {num}".format(num=assert_per))
    except Exception as e:
        print(e)
    return (arrange_per,action_per,assert_per)


#%%

data_edit = Data('./all Projects/Data_NoLevel/','edit distance', 'basic', tag=True, data_structure='clean').df
print(data_edit.shape)
#%%
data_Jacard = Data('./all Projects/Data_NoLevel/','jacard', 'basic', tag=True, data_structure='clean').df
#%%
ACTION_LIMITS_IN_ONE_CASE = 2
ASERT_DISTANCE_LIMITS = 3


baseline = Baseline(data_edit)
# baseline = Baseline(data_Jacard)
# baseline.name_match_distribution()
# action_distribution, all_distribution, name_sim_distribution, assert_dis_distribution, AR_AS_distribution = baseline.case_rank_with_edit_distance_or_Jcard(distribution_show=True)
#%%
prob = baseline.model(action_limits_in_one_case=ACTION_LIMITS_IN_ONE_CASE,assert_distance_limits=ASERT_DISTANCE_LIMITS)
target = data_edit['AAA'].values

arrange_recall,action_recall,assert_recall =recall_kf(prob,target)
arrange_pre,action_pre,assert_pre = percision_kf(prob,target)



# print(pred)


# %%
def f1(p,r):
    return 2*p*r/(p+r)

# print(f'ONE CASE WILL HAVE {ACTION_LIMITS_IN_ONE_CASE} ACTIONS, AND ONLY THE STATEMENTS WITH ASSERT DISTANCE EQUAL OR LESS THAN {ASERT_DISTANCE_LIMITS} WILL BE CONSIDERED AS ACTION')
print(f"arrange_pre: {arrange_pre}, arrange_recall: {arrange_recall}, arrange_f1: {f1(arrange_pre,arrange_recall)}")
print(f"action_pre: {action_pre}, action_recall: {action_recall}, action_f1: {f1(action_pre,action_recall)}")
print(f"assert_pre: {assert_pre}, assert_recall: {assert_recall}, assert_f1: {f1(assert_pre,assert_recall)}")
# %%
data_edit.tail()
# %%
pd.DataFrame({"predict":prob,"target":target}).to_csv('./temp.csv')
# %%
data_edit.to_csv('./temp_all.csv')
# %%
import plotly.express as px
import plotly.graph_objects as go

fig00 = px.histogram(action_distribution,nbins=100)
fig00.show()
fig0 = px.histogram(AR_AS_distribution,nbins=100)
fig0.show()

fig = go.Figure()
fig.add_trace(go.Histogram(x=action_distribution,name="action distribution"))
fig.add_trace(go.Histogram(x=AR_AS_distribution,name="AR_AS distribution"))
# Overlay both histograms
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
fig.show()

fig2 = px.histogram(all_distribution,title="all distribution")
fig2.show()
fig3 = px.histogram(name_sim_distribution,title="name_sim distribution")
fig3.show()
fig4 = px.histogram(assert_dis_distribution,title="assert_dis distribution")
fig4.show()
# %%
data_Jacard.head()
# %%
li = [1,2,3,4,5,7,10]
from sklearn import preprocessing
import numpy as np
x_array = np.array(li)
normalized_arr = preprocessing.normalize([x_array])
print(normalized_arr)

# %%
import sklearn.preprocessing
scaler = sklearn.preprocessing.MinMaxScaler()
a = data_edit["Name Similarity"][0:2].values
print(a)
# %%
data_edit[3950:3960]

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
# fig.add_trace(
#     go.Scatter(x=[1, 2, 3], y=[40, 50, 60], name="yaxis data"),
#     secondary_y=False,
# )
fig.add_trace(go.Histogram(x=action_distribution,name="Action Statements",nbinsx=100),secondary_y=False)
fig.add_trace(go.Histogram(x=AR_AS_distribution,name="Non-Action Statements",nbinsx=100),secondary_y=True)

# fig.add_trace(
#     go.Scatter(x=[2, 3, 4], y=[4, 5, 6], name="yaxis2 data"),
#     secondary_y=True,
# )

# Add figure title
fig.update_layout(
    title_text="Threashold distribution",
)
fig.update_layout(barmode='overlay')
# Reduce opacity to see both histograms
fig.update_traces(opacity=0.75)
# Set x-axis title
fig.update_xaxes(title_text="Action Reletive Score")

# Set y-axes titles
fig.update_yaxes(title_text="Action Statement Quantity", secondary_y=False)
fig.update_yaxes(title_text="Non-Action Statement Quantity", secondary_y=True)

fig.show()
# %%
