import os
import numpy as np

def findAllFile(base):
    '''
    generator function
        Usage: for file in findAllFile(path):
                ...
        Walk through all the files in the path, 
        no matter how deep the file is hide. 

        It won't include folder as final result. 
    '''
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def Data_twist(df):
    '''
    Remove all the conflict data in the AAA tagging.
    '''
    df = df.fillna(0)
    for i in range(df.shape[0]):
        if df.loc[i,'AAA'] == '1,2':
            df.loc[i,'AAA'] = 2
        elif df.loc[i,'AAA'] == '0,1':
            df.loc[i,'AAA'] = 0
        elif df.loc[i,'AAA'] == '0,2':
            df.loc[i,'AAA'] = 0
        elif df.loc[i,'AAA'] == '0,1,2':
            df.loc[i,'AAA'] = 0
        elif df.loc[i,'AAA'] =='/':
            df.loc[i,'AAA'] = 0
        elif df.loc[i,'AAA'] == np.nan:
            df.loc[i,'AAA'] = 0
    return df

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