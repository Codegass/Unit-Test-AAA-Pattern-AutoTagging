import pandas as pd
from util import findAllFile, Data_twist

def reStructureData(input_df):
    '''
    pd.DataFrame -> pd.DataFrame

    Clean the dataframe from multiple level tag file 
    to statement level tag 
    with only AAA tag reamins.
    '''
    testClassName = input_df['Contains Assertion'][3:].rename('testClassName')
    testMethodName = input_df['Comment/Javadoc is meanningful'][3:].rename('testMethodName')
    potentialTargetQualifiedName = input_df['Naming Convention is meanningful'][3:].rename('potentialTargetQualifiedName')
    AAA = input_df['Test target related to dynamic binding'][3:].rename('AAA')
    result_df = pd.concat([testClassName, testMethodName, potentialTargetQualifiedName, AAA], axis=1)
    return Data_twist(result_df.reset_index(drop=True))

def count_tab(input_str):
    '''
    count how many space will be in the beggining of the string.
    And divide it with 5 to have the tab number as the level info
    '''
    space_count = 0
    for character in input_str:
        if character == ' ':
            space_count += 1
        else:
            break
    # print(space_count)
    return space_count/5

def statementCleanBasedOnLevel(input_df):
    '''
    pd.DataFrame -> pd.DataFrame

    clean all the statement based on levels.
    Only keep the lowest level of the statements.
    '''
    result_df = pd.DataFrame(columns=['testClassName','testMethodName','potentialTargetQualifiedName','AAA'])
    for i in range(input_df.shape[0]): # get the col numbers
        col_df = input_df.iloc[[i]] # double the [] will give the col become the dataframe, if only use one [], the col will be pd.series
        if i+1 >= input_df.shape[0]:
            result_df = pd.concat([result_df,col_df],ignore_index=True)
        else:
            # print(col_df['potentialTargetQualifiedName'])
            current_line_tab = count_tab(col_df['potentialTargetQualifiedName'].values[0])
            next_line_tab = count_tab(input_df.iloc[[i+1]]['potentialTargetQualifiedName'].values[0])
            if current_line_tab >= next_line_tab:
                # TODO:Add Log here
                result_df = pd.concat([result_df,col_df],ignore_index=True)
    return result_df

def buildNewFileName(file_path):
    name_list = file_path.split('/')[-1].split('.')[0:2]
    name = file_path.split('/')[-2]+'/'+name_list[0]+'.'+name_list[1]+'.csv'
    return name

def dataLevelRemoving(path):
    for file in findAllFile(path):
        try:
            # try will avoid the the pronlems like .DS_store
            df = pd.read_excel(file) # read the data from file as DF
            df = reStructureData(df) # clean the data
            df = statementCleanBasedOnLevel(df) # only keep the sub 
            # save
            name = buildNewFileName(file)
            df.to_csv('./all Projects/Data_NoLevel/'+name,index=False)
        except Exception as e:
            print(file)
            print(e)


if __name__ == '__main__':
    df = dataLevelRemoving('./all Projects/P-Merged/') # this path is just for test 

