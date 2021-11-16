import numpy as np
import pandas as pd
import re, os
from sklearn.metrics import cohen_kappa_score

class Data:
    
    def __init__(self, folder_path, nameSimilarityMethod, assertDistanceMethod, tag=False, Word2Vec='SO'):

        self.nameSimilarityMethod = nameSimilarityMethod
        self.assertDistanceMethod = assertDistanceMethod
        self.base = folder_path
        self.all_file_path_list = []
        self.word2vec_flag = Word2Vec
        self.df = pd.DataFrame(columns=['testClassName','testMethodName','potentialTargetQualifiedName','AAA'])
        self.tag = tag
        # if self.nameSimilarityMethod == 'word2vec':
        #     if self.word2vec_flag == 'SO':
        #         self.word2vec_model = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
        #     if self.word2vec_flag == 'WK':
        #         # self.word2vec_model = Word2Vec(api.load('wiki-english-20171001'))
        #         corpus = api.load('text8')
        #         self.word2vec_model = W2V(corpus)
        # else:
        #     self.word2vec_model = None
        
        for i in self.findAllFile():
            if i == './alln/.DS_Store': #TODO: need fix
                continue
            else:
                self.all_file_path_list.append(i)
        
        self.dataframe_clean_and_merge()
        # self.build_new_features()

    def findAllFile(self):
        for root, ds, fs in os.walk(self.base):
            for f in fs:
                fullname = os.path.join(root, f)
                yield fullname
    
    def remove_null(self, strlist):
        result_list = []
        for _ in strlist:
            if _ != '':
                result_list.append(_)
        return result_list

    def remove_all_punctuantion(self, method):
        if type(method) == str:
            return method.split('_')
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + _.split('_')
            return self.remove_null(result_list)

    def convert_2_to_to(self, method):
        if type(method) == str:
            return method.replace('2','to')
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list.append(_.replace('2','to'))
            return self.remove_null(result_list)

    def remove_word_test(self, method):
        if type(method) == str:
            return self.remove_null(method.split('test'))
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + _.split('test')
            return self.remove_null(result_list)

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
    
    def lower_case(self, method):
        if type(method) == str:
            return method.lower()
        elif type(method) == list:
            result_list = list()
            for _ in method:
                result_list.append(_.lower())
            return self.remove_null(result_list)

    def remove_all_stop_words(self, method):
        if type(method) == str:
            return method
        elif type(method) == list:
            filtered_words = [word for word in method if word not in stopwords.words('english')]
            return filtered_words

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

    def specific_twist(self, method):
        if type(method) == str:
            result_list = re.sub( r"(\d+)", r" \1 ", method).split()
            return result_list
        if type(method) == list:
            result_list = list()
            for _ in method:
                result_list = result_list + re.sub( r"(\d+)", r" \1 ", _).split()
            return self.remove_null(result_list)

    
        

    def test_name_process(self,method):
        # if self.nameSimilarityMethod == 'word2vec':
        #     return self.remove_all_stop_words(self.lower_case(self.remove_word_test(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(method))))))
        # else:
        return self.specific_twist(self.remove_all_stop_words(self.lower_case(self.remove_word_test(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(method)))))))
    
    def potential_method_name_process(self,method):
        # print(re.findall(r"(\w+)\<", re.split('\\.',method)[-1]))
        # print(re.findall(r"(\w+)\(", re.split('\\.',method)[-1]))
        # print(method)
        
        try: 
            # short_method_name = re.findall(r"(\w+)\<", re.split('\\.',method)[-1])[0]
            # print('----')
            # print(method)
            # print('____')
            print(method)
            short_method_name = re.findall(r"(\w+)\(", method)[0]
        except:
            # short_method_name = re.findall(r"(\w+)\(", re.split('\\.',method)[-1])[0]
            try:
                # print(method)
                str_method = str(method)
                short_method_name = re.findall(r"(\w+)\<", str_method)[0]
            except Exception as e2:
                short_method_name = method
                print(method)
                print(e2)
            
        return self.specific_twist(self.remove_all_stop_words(self.lower_case(self.remove_word_test(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(short_method_name)))))))

    def dataframe_clean_and_merge(self):
        count = 0
        for file in self.all_file_path_list:
            # read the file
            try:
                raw_df = pd.read_excel(file)
            except Exception as e:
                print(e)
                print(file)
                continue
            
            #build the new df from the raw one

            testClassName = raw_df['Contains Assertion'][3:].rename('testClassName')
            testMethodName = raw_df['Comment/Javadoc is meanningful'][3:].rename('testMethodName')
            potentialTargetQualifiedName = raw_df['Naming Convention is meanningful'][3:].rename('potentialTargetQualifiedName')
            AAA = raw_df['Test target related to dynamic binding'][3:].rename('AAA')
            New_df = pd.concat([testClassName, testMethodName, potentialTargetQualifiedName, AAA], axis=1)
            # print(New_df)
            self.df = pd.concat([self.df,New_df],ignore_index=True)
            # print(self.df)
            count += 1
        print("Data Merge Finished. {Num} files are merged.".format(Num = count))
        self.df['Assert Distance'] = 0.0
        self.df['Name Similarity'] = 0.0
        self.df['Level'] = 0.0
        if self.tag:
            self.df['Tag-Mock'] = 0
            self.df['Tag-New'] = 0
            self.df['Tag-Test'] = 0
            self.df['Tag-Get'] = 0
            self.df['Tag-Set'] = 0


    def Data_twist(self,df):
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

def filter(input_list,target):
    result_list = list()
    for _ in input_list:
        if _ == target:
            result_list.append(_)
        else:
            result_list.append(9)
    if len(result_list) == len(input_list):
        return result_list
    else:
        print('len error')
        return result_list   

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def Data_twist(df):
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


if __name__ == "__main__":
    df_p1 = Data('./all Projects/P1-N/','edit distance','basic',tag=True).df
    df_p2 = Data('./all Projects/P3-W/','edit distance','basic',tag=True).df
    kappa_list_p1 = Data_twist(df_p1)['AAA'].values.tolist()
    kappa_list_p2 = Data_twist(df_p2)['AAA'].values.tolist()
    kappa_noTwi_p1 = df_p1['AAA'].values.tolist()
    ### Whole Kappa ###
    print('whole KAPPA: {}'.format(cohen_kappa_score(kappa_list_p2,kappa_list_p1)))
    ### AR KAPPA ###
    kappa_AR_p1 = filter(kappa_list_p1,0)
    kappa_AR_p2 = filter(kappa_list_p2,0)
    ar_kappa = cohen_kappa_score(kappa_AR_p1,kappa_AR_p2)
    print('AR KAPPA: {}'.format(ar_kappa))
    ### AC KAPPA ###
    kappa_AC_p1 = filter(kappa_list_p1,1)
    kappa_AC_p2 = filter(kappa_list_p2,1)
    ac_kappa = cohen_kappa_score(kappa_AC_p1,kappa_AC_p2)
    print('AC KAPPA: {}'.format(ac_kappa))
    ### AS KAPPA ###
    kappa_AS_p1 = filter(kappa_list_p1,2)
    kappa_AS_p2 = filter(kappa_list_p2,2)
    as_kappa = cohen_kappa_score(kappa_AS_p1,kappa_AS_p2)
    print('AS KAPPA: {}'.format(as_kappa))
    ## No Twisted Kappa ##
    kappa_notw_AC_p1 = []
    for i in kappa_noTwi_p1:
        if type(i) == str:
            kappa_notw_AC_p1.append(9)
        elif i != 1:
            kappa_notw_AC_p1.append(9)
        else:
            kappa_notw_AC_p1.append(1)
    print(cohen_kappa_score(kappa_notw_AC_p1,kappa_AC_p2))

