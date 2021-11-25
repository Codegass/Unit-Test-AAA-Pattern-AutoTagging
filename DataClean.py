import re, os
import distance
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg.linalg import _assert_stacked_square
import sklearn as skl
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from scipy.linalg import norm
from gensim.models import Word2Vec as W2V
import gensim.downloader as api
from gensim.models.keyedvectors import KeyedVectors

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class nameSimilarity:
    '''
    All the methods for the calculation of the Name similarity between class&Case name and method name.
    the input show be the 
    '''
    def __init__(self,testMethod,caseName,useWord2Vec,word2vec_model):
        # self.measureMethod = measureMethod
        self.testMethodName = testMethod
        self.caseName = caseName # including the class and casemethod name
        if useWord2Vec:
            self.word_vec = word2vec_model
            # api.info('text8')
            # dataset = api.load('text8')
            # # sentences = word2vec.Text8Corpus('text8')
            # self.model = Word2Vec(dataset,min_count=1)
    
    def list_to_str(self,input_list):
        merged_str = ''
        for i in range(len(input_list)):
            if i != len(input_list):
                merged_str = merged_str + input_list[i] + ' '
            else:
                merged_str = merged_str + input_list[i]
        return merged_str

    def editDistance(self):
        '''can handle list'''
        return distance.levenshtein(self.testMethodName,self.caseName)

    def jacard_similarity(self):
        # print(self.testMethodName)
        # print(self.caseName)
        # print('****')
        s1= self.list_to_str(self.testMethodName)
        s2= self.list_to_str(self.caseName)
        # convert to TF Matrix
        cv = skl.feature_extraction.text.CountVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1,s2]
        # print(s1)
        # print(s2)
        # print('_____')
        vectors = cv.fit_transform(corpus).toarray()
        #intersection
        numerator = np.sum(np.min(vectors,axis=0))
        #Union
        denominator = np.sum(np.max(vectors,axis=0))
        #Jacard
        return 1.0 * numerator/denominator
    
    def cosine_similarity(self,x,y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def Word2Vec_SO(self): 
        '''
        Use the Stack over Flow dic for Word2Vec.
        Here we use the simpliest way average to calculate the sentence vector
        '''
        casenameVec = np.zeros(200)
        testMethodNameVec = np.zeros(200)
        case_count = 0
        for word in self.caseName:
            case_count += 1
            try:
                vec = self.word_vec.get_vector(word)
                # print('word:{}'.format(word))
                # print('vec:{}'.format(vec))
            except Exception as e:
                vec = np.zeros(200)
                # print(e)
            casenameVec = casenameVec+vec

        AVG_case_Vec = casenameVec/case_count
        # print('casenameVec:{}'.format(AVG_Vec))

        method_count = 0
        for word in self.testMethodName:
            method_count += 1
            try:
                vec = self.word_vec.get_vector(word)
            except Exception as e:
                vec = np.zeros(200)
                # print(e)
            testMethodNameVec = testMethodNameVec +vec
        AVG_testMethodNameVec = testMethodNameVec/method_count

        result_similarity = self.cosine_similarity(AVG_case_Vec,AVG_testMethodNameVec)
        if not result_similarity:
            print('case name is {}'.format(self.caseName))
            print("AVG_case_VEC:")
            print(AVG_case_Vec)
            print("AVG_testMethodNameVec:")
            print(AVG_testMethodNameVec)
        return result_similarity

    def Word2Vec_WK(self): 
        '''
        Use the Stack over Flow dic for Word2Vec.
        Here we use the simpliest way average to calculate the sentence vector
        '''
        casenameVec = np.zeros(100)
        testMethodNameVec = np.zeros(100)
        case_count = 0
        for word in self.caseName:
            case_count += 1
            try:
                vec = self.word_vec.wv[word]
            except Exception as e:
                vec = np.zeros(100)
                print(e)
            casenameVec = casenameVec+vec

        AVG_case_Vec = casenameVec/case_count

        method_count = 0
        for word in self.testMethodName:
            method_count += 1
            try:
                vec = self.word_vec.wv[word]
            except Exception as e:
                vec = np.zeros(100)
                print(e)
            testMethodNameVec = testMethodNameVec +vec
        AVG_testMethodNameVec = testMethodNameVec/method_count
        # print('Similarity:')
        result_similarity = self.cosine_similarity(AVG_case_Vec,AVG_testMethodNameVec)
        if not result_similarity:
            print('case name is {}'.format(self.caseName))
            print("AVG_case_VEC:")
            print(AVG_case_Vec)
            print("AVG_testMethodNameVec:")
            print(AVG_testMethodNameVec)
        return result_similarity

class Data:
    
    def __init__(self, folder_path, nameSimilarityMethod, assertDistanceMethod, tag=False, Word2Vec='SO', data_structure='raw'):

        self.nameSimilarityMethod = nameSimilarityMethod
        self.assertDistanceMethod = assertDistanceMethod
        self.base = folder_path
        self.all_file_path_list = []
        self.word2vec_flag = Word2Vec
        self.df = pd.DataFrame(columns=['testClassName','testMethodName','potentialTargetQualifiedName','AAA'])
        self.tag = tag
        self.data_structure = data_structure
        if self.nameSimilarityMethod == 'tfidf': #tfidf special
            for i in self.findAllFile():
                if i[-8:] == 'DS_Store': #TODO: need fix
                    continue
                else:
                    self.all_file_path_list.append(i)

            self.df = self.all_project_tfidf_with_other_feature()
        else: # common methods
            if self.nameSimilarityMethod == 'word2vec':
                if self.word2vec_flag == 'SO':
                    
                    self.word2vec_model = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)
                if self.word2vec_flag == 'WK':
                    # self.word2vec_model = Word2Vec(api.load('wiki-english-20171001'))
                    corpus = api.load('text8')
                    self.word2vec_model = W2V(corpus)
            else:
                self.word2vec_model = None
            
            for i in self.findAllFile():
                if i[-8:] == 'DS_Store': #TODO: need fix
                    continue
                else:
                    self.all_file_path_list.append(i)
            
            self.dataframe_clean_and_merge()
            self.build_new_features()

    def findAllFile(self,path=None):
        if path == None:
            for root, ds, fs in os.walk(self.base):
                for f in fs:
                    fullname = os.path.join(root, f)
                    yield fullname
        else:
            for root, ds, fs in os.walk(path):
                for f in fs:
                    fullname = os.path.join(root, f)
                    yield fullname
        
    def cosine_similarity(self,x,y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

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

        return self.specific_twist(self.remove_all_stop_words(self.lower_case(self.remove_word_test(self.remove_all_punctuantion(self.IT_split(self.split_with_upper_case(method)))))))
    
    def potential_method_name_process(self,method):
        
        try: 
            short_method_name = re.findall(r"(\w+)\(", method)[0]
        except:
            try:
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
                if file.split('.')[-1] == 'csv':
                    raw_df = pd.read_csv(file)
                elif file.split('.')[-1] == 'xlsx':
                    raw_df = pd.read_excel(file)
                elif file.split('.')[-1] == 'DS_Store':
                    continue
                else:
                    print('file name is wrong type to handle : {}'.format(file))
                    break
            except Exception as e:
                print(e)
                print(file)
                continue
            
            #build the new df from the raw one
            if self.data_structure == 'raw':
                testClassName = raw_df['Contains Assertion'][3:].rename('testClassName')
                testMethodName = raw_df['Comment/Javadoc is meanningful'][3:].rename('testMethodName')
                potentialTargetQualifiedName = raw_df['Naming Convention is meanningful'][3:].rename('potentialTargetQualifiedName')
                AAA = raw_df['Test target related to dynamic binding'][3:].rename('AAA')
                New_df = pd.concat([testClassName, testMethodName, potentialTargetQualifiedName, AAA], axis=1)
            elif self.data_structure == 'clean':
                New_df = raw_df
            else:
                print('data sturture type setting is missing')
            # print(New_df)
            self.df = pd.concat([self.df,New_df],ignore_index=True)
            # print(self.df)
            count += 1
        print("Data Merge Finished.".format(Num = count))
        ## fill all the blank part features
        self.df['Assert Distance'] = 0.0
        self.df['Name Similarity'] = 0.0
        # self.df['Level'] = 0.0
        if self.tag:
            self.df['Tag-Mock'] = 0
            self.df['Tag-New'] = 0
            self.df['Tag-Test'] = 0
            self.df['Tag-Get'] = 0
            self.df['Tag-Set'] = 0

    def dataframe_clean_and_build_seq(self):
        '''
        NOT USED RIGHT NOW
        '''
        count = 0
        res_list = []
        re_target = []
        for file in self.all_file_path_list:
            # read the file
            try:
                raw_df = pd.read_excel(file)
            except Exception as e:
                print(e)
                print(file)
            
            #build the new df from the raw one

            testClassName = raw_df['Contains Assertion'][3:].rename('testClassName')
            testMethodName = raw_df['Comment/Javadoc is meanningful'][3:].rename('testMethodName')
            potentialTargetQualifiedName = raw_df['Naming Convention is meanningful'][3:].rename('potentialTargetQualifiedName')
            AAA = raw_df['Test target related to dynamic binding'][3:].rename('AAA')
            New_df = pd.concat([testClassName, testMethodName, potentialTargetQualifiedName, AAA], axis=1)
            New_df.reset_index(drop=True, inplace=True)
            # print(New_df)
            for i in range(New_df.shape[0]):

                # get the name data
                testClassName = self.test_name_process(New_df['testClassName'][i])
                testMethodName = self.test_name_process(New_df['testMethodName'][i])
                try:
                    testName = testClassName + testMethodName
                except Exception as e:
                    print(e)
                    # print(testMethodName)
                    # print(testClassName)
                    print(New_df.iloc[i-1])
                    print('test name list merge issue. at {i}'.format(i=i))
                # print(New_df.iloc[i])
                potentialTargetQualifiedName = self.potential_method_name_process(New_df['potentialTargetQualifiedName'][i])

                # build the name similarity
                if self.nameSimilarityMethod == 'edit distance':
                    New_df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,False).editDistance()
                    # print('testName is ', testName)
                    # print(potentialTargetQualifiedName)
                    # print(New_df.loc[i,'Name Similarity'])
                elif self.nameSimilarityMethod == 'jacard':
                    New_df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,False).jacard_similarity()
                elif self.nameSimilarityMethod == 'word2vec':
                    New_df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,True).Word2Vec()
                else:
                    print('Wrong name similarity method name')

                # build assert distance
                if self.assertDistanceMethod == 'basic':
                    assert_dis = 0
                    casename = New_df['testMethodName'][i]
                    while assert_dis < (New_df.shape[0]-i):
                        if casename != New_df['testMethodName'][(i+assert_dis)]:
                            assert_dis = -1 #TODO:
                            break
                        if re.search(r'ASSERT',New_df['potentialTargetQualifiedName'][(i+assert_dis)]) :
                            break
                        else:
                            assert_dis += 1

                    New_df.loc[i,'Assert Distance'] = assert_dis

                elif self.assertDistanceMethod == 'advance':
                    pass #TODO:
                else:
                    print('Wrong assert distance method name')


                # # build level
                # space_count = 0
                # for character in New_df['potentialTargetQualifiedName'][i]:
                #     if character == ' ':
                #         space_count += 1
                #     else:
                #         break
                # New_df.loc[i, 'Level'] = space_count

                # drop other 
                New_df = self.Data_twist(New_df)

            res_list.append(np.array(New_df.drop(labels=['testClassName','testMethodName','potentialTargetQualifiedName','AAA'], axis=1)))
            # add start and end mark
            target_seq = np.append(np.insert(New_df['AAA'].values,0,3),4)
            re_target.append(target_seq)
        return (res_list,re_target)

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

    def build_new_features(self):

    
        count = 0
        for i in range(self.df.shape[0]):

            # get the name data
            testClassName = self.test_name_process(self.df['testClassName'][i])
            testMethodName = self.test_name_process(self.df['testMethodName'][i])
            try:
                testName = testClassName + testMethodName
                # print(testMethodName)
                # print(testClassName)
            except Exception as e:
                print(e)
                # print(testMethodName)
                # print(testClassName)
                print(self.df.iloc[i-1])
                print('test name list merge issue. at {i}'.format(i=i))
            # print(self.df.iloc[i])
            potentialTargetQualifiedName = self.potential_method_name_process(self.df['potentialTargetQualifiedName'][i])
            # print(self.df['potentialTargetQualifiedName'][i])
            # print(potentialTargetQualifiedName)
            # print('_____')

            # build the name similarity
            if self.nameSimilarityMethod == 'edit distance':
                self.df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,False,self.word2vec_model).editDistance()
                # print('testName is ', testName)
                # print(potentialTargetQualifiedName)
                # print(self.df.loc[i,'Name Similarity'])
            elif self.nameSimilarityMethod == 'jacard':
                self.df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,False,self.word2vec_model).jacard_similarity()
            elif self.nameSimilarityMethod == 'word2vec' and self.word2vec_flag == 'SO':
                self.df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,True,self.word2vec_model).Word2Vec_SO()
            elif self.nameSimilarityMethod == 'word2vec' and self.word2vec_flag == 'WK':
                self.df.loc[i,'Name Similarity'] = nameSimilarity(testName,potentialTargetQualifiedName,True,self.word2vec_model).Word2Vec_WK()
            else:
                print('Wrong name similarity method name')

            # build assert distance
            if self.assertDistanceMethod == 'basic':
                assert_dis = 0
                casename = self.df['testMethodName'][i]
                while assert_dis < (self.df.shape[0]-i):
                    if casename != self.df['testMethodName'][(i+assert_dis)]:
                        assert_dis = -1 #TODO:
                        break
                    if re.search(r'ASSERT',self.df['potentialTargetQualifiedName'][(i+assert_dis)]) :
                        break
                    else:
                        assert_dis += 1

                self.df.loc[i,'Assert Distance'] = assert_dis

            # build Tag features
            if self.tag:
                try: 
                    tag_keyword = self.df['potentialTargetQualifiedName'][i].lstrip().split(' ')[0]
                except Exception as e:
                    print(e)
                    print('{0} line has no 0 position'.format(i))
                if tag_keyword == 'MOCK':
                    # print("Mock")
                    # print(tag_keyword)
                    self.df.loc[i, 'Tag-Mock'] = 1
                elif tag_keyword == 'NEW':
                    # print("New")
                    # print(tag_keyword)
                    self.df.loc[i, 'Tag-New'] = 1
                elif tag_keyword == 'TEST':
                    # print("Test")
                    # print(tag_keyword)
                    self.df.loc[i, 'Tag-Test'] = 1

                pattern_1 = re.compile('.get')
                pattern_2 = re.compile('.set')
                str= self.df['potentialTargetQualifiedName'][i].lstrip()
                m_1 = pattern_1.search(str, re.I)
                m_2 = pattern_2.search(str, re.I)
                if m_1 != None:
                    self.df.loc[i, 'Tag-Get'] = 1
                elif m_2 != None:
                    self.df.loc[i, 'Tag-Set'] =1

######### TFIDF ##############
    def Project_summary(self, path):
        '''
        Collect all the class names, interface names and method names.

        Path -> collect all the class names, interface names and method names and put into one list.
            -> put all the file lists into one list.

        Parameters
        ----------
        path : str
            The path of the project.

        Returns
        -------
        all_files: list
            A list contains all the files' class/interface/method names' list.
        '''
        # one file is one document
        all_files = list()
        # print('Collecting all the files for the project...')
        for file in self.findAllFile(path):
            # fix the issue for the .DS_Store file
            if file[-4:] != 'java':
                continue
            with open(file,'r') as f: 
                file_str = f.read()
                summary = list()
                for line in file_str.split('\n'):
                    # find all the public or private or protected lines
                    if re.search('public',line) or re.search('private',line) or re.search('protected',line):
                        # find all the class or interface
                        if re.search('class', line) or re.search('interface', line):
                            line_list = line.split(' ')
                            for i in range(len(line_list)):
                                if line_list[i] == 'class':
                                    try:
                                        summary.append(line_list[i+1])
                                    except Exception as e:
                                        logging.debug(e)
                                        logging.debug(line_list)
                                        logging.debug('This case is using \\n after the class keyword')
                                elif line_list[i] == 'interface':
                                    try:
                                        summary.append(line_list[i+1])
                                    except Exception as e:
                                        logging.debug(e)
                                        logging.debug(line_list)
                                        logging.debug('this file is using \\n after the interface keyword')
                                elif line_list[i] == 'implements':
                                    try:
                                        if re.findall('\w+\<',line_list[i+1]):
                                            summary.append(re.findall('\w+\<',line_list[i+1])[0][:-1])
                                    except Exception as e:
                                        logging.debug(e)
                                        logging.debug(line_list)
                                        logging.debug('this file is using \\n after the implements keyword')
                                elif line_list[i] == 'extends':
                                    try:
                                        summary.append(line_list[i+1])
                                    except Exception as e:
                                        logging.debug(e)
                                        logging.debug(line_list)
                                        logging.debug('this file is using \\n after the extends keyword')
                        else:
                        # find all the method
                            function = re.findall('\w+\(',line)
                            if function:
                                for i in function:
                                    summary.append(i[:-1])
                splitted_summary = list()
                for i in summary:
                    splitted_summary += self.test_name_process(i)
                all_files.append(' '.join(summary))
        # print('Collecting all the files for the project finished!')
        return all_files

    def one_project_tfidf(self, project_path_list):
        '''
        Build the tf-idf matrix for the project.

        Parameters
        ----------
        path : str
            The path of the project.

        Returns
        -------
        one_project_df: DataFrame
            A DataFrame contains the all the features for the project.
        '''
        # get all the keywords in one project and build as the string list.
        # one string is one document
        all_files = list()
        temp_store_list = list()
        df_list = list()
        one_project_df = pd.DataFrame(columns=['testClassName','testMethodName','potentialTargetQualifiedName','AAA','Name Similarity','Assert Distance','Tag-Mock','Tag-New','Tag-Test','Tag-Get','Tag-Set'])
        print('Calculating TFIDF for the project...')
        for path in project_path_list:
            # print(path)
            all_files += self.Project_summary(path)
        # we need to prepare the location info in the list for the tfidf
        start_point = len(all_files)

        # read the test statements file
        for file in self.all_file_path_list:
            # read the file
            try:
                if file.split('.')[-1] == 'csv':
                    raw_df = pd.read_csv(file)
                elif file.split('.')[-1] == 'xlsx':
                    raw_df = pd.read_excel(file)
                elif file.split('.')[-1] == 'DS_Store':
                    continue
                else:
                    print('file name is wrong type to handle : {}'.format(file))
                    break
            except Exception as e:
                print(e)
                print(file)
                continue
            # this function only cosider the level removed files.
            # TODO: add the method to handle the raw excel file.
        

            ## deal with one case
            New_df = raw_df
            ## fill all the blank part features
            New_df['Assert Distance'] = 0.0
            New_df['Name Similarity'] = 0.0
            if self.tag:
                New_df['Tag-Mock'] = 0
                New_df['Tag-New'] = 0
                New_df['Tag-Test'] = 0
                New_df['Tag-Get'] = 0
                New_df['Tag-Set'] = 0
            # build the case string
            case_str = list()
            case_str += self.test_name_process(New_df['testClassName'][0])
            case_str += self.test_name_process(New_df['testMethodName'][0])
            for string in New_df['potentialTargetQualifiedName']:
                case_str += self.potential_method_name_process(string)
            temp_store_list.append(case_str)
            case_str = ' '.join(case_str)
            # add case string to the all_files list
            all_files.append(case_str)
            # add the original dataframe to the df_list
            df_list.append(New_df)
            

        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(all_files).toarray()
        words_dic = vectorizer.vocabulary_
        # with open('words_dic.txt','wb') as f:
        #     f.write(str(words_dic).encode())
        # build name similarity based on TFIDF
        for i in range(len(df_list)):
            df = df_list[i]
            case_matrix = tfidf_matrix[start_point+i]

            # build the test case&class name vector
            test_case_and_class_name_vector = np.zeros(len(words_dic))
            
            test_class_name_list = self.test_name_process(df['testClassName'][0])
            test_case_name_list = self.test_name_process(df['testMethodName'][0])
            test_case_and_class_name_list = list(set(test_class_name_list + test_case_name_list))

            for word in test_case_and_class_name_list:
                if word in words_dic:
                    test_case_and_class_name_vector[words_dic[word]] = case_matrix[words_dic[word]]
            
            # build the potential target vector
            for i in range(df.shape[0]):
                potential_target_vector = np.zeros(len(words_dic))
                try:
                    potential_target_list = temp_store_list[2+i]
                except Exception as e:
                    print(e)
                    print(all_files[start_point+i])
                    print(temp_store_list[i])
                    continue
                for word in potential_target_list:
                    if word in words_dic:
                        potential_target_vector[words_dic[word]] = case_matrix[words_dic[word]]
                # calculate the similarity
                try:
                    case_and_class_name_np = np.array([test_case_and_class_name_vector])
                    potential_target_np = np.array([potential_target_vector])
                    similarity = cosine_similarity(case_and_class_name_np, potential_target_np)[0]
                    if not similarity:
                        logging.debug(case_and_class_name_np)
                        logging.debug(test_case_and_class_name_list)
                        logging.debug(potential_target_np)
                        logging.debug(potential_target_list)
                        similarity = 0.0
                except Exception as e:
                    print(e)
                    logging.debug(test_case_and_class_name_vector)
                    logging.debug(test_case_and_class_name_list)
                    logging.debug(potential_target_vector)
                    logging.debug(potential_target_list)
                    print(all_files[start_point+i])
                df.loc[i,'Name Similarity'] = similarity
            
            one_project_df = pd.concat([one_project_df,df],ignore_index=True)
        print('Calculating TFIDF for the project finished!')
        return one_project_df

    def all_project_tfidf_with_other_feature(self):
        #TODO: fix the hardcoded path prefix
        accomulo_project_path_list = [
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/core/src/test/java/org/apache/accumulo/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/hadoop-mapreduce/src/test/java/org/apache/accumulo/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/minicluster/src/test/java/org/apache/accumulo/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/base/src/test/java/org/apache/accumulo/server/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/gc/src/test/java/org/apache/accumulo/gc/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/master/src/test/java/org/apache/accumulo/master/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/monitor/src/test/java/org/apache/accumulo/monitor/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/tracer/src/test/java/org/apache/accumulo/tracer/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/server/tserver/src/test/java/org/apache/accumulo/tserver',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/shell/src/test/java/org/apache/accumulo/shell/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/start/src/test/java/org/apache/accumulo/start/',
            '/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/test/src/main/java/org/apache/accumulo/'
        ]
        
        druid_project_path_list = [
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/benchmarks/src/test/java/org/apache/druid/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/cloud/aws-common/src/test/java/org/apache/druid/common/aws/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/cloud/gcp-common/src/test/java/org/apache/druid/common/gcp/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/core/src/test/java/org/apache/druid/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/extendedset/src/test/java/org/apache/druid/extendedset/intset/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/hll/src/test/java/org/apache/druid/hll/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/indexing-hadoop/src/test/java/org/apache/druid/indexer/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/indexing-service/src/test/java/org/apache/druid/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/integration-tests/src/test/java/org/apache/druid/tests/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/processing/src/test/java/org/apache/druid/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/server/src/test/java/org/apache/druid/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/services/src/test/java/org/apache/druid/cli/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/sql/src/test/java/org/apache/druid/sql/'
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/extensions-contrib/',
            '/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/extensions-core/'
        ]

        cloudstack_project_path_list = [
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/agent/src/test/java/com/cloud/agent/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/api/src/test/java/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/core/src/test/java/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/engine/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/framework/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/plugins/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/server/src/test/java/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/services/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/usage/',
            '/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/utils/src/test/java/'
        ]

        dubbo_project_path_list = [
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-cluster/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-common/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-compatible/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-config/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-configcenter/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-container/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-dependencies/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-filter/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-metadata/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-monitor/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-plugin/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-registry/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-remoting/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-rpc/',
            '/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/dubbo-serialization/'
        ]

        projects = [accomulo_project_path_list, druid_project_path_list, cloudstack_project_path_list, dubbo_project_path_list]
        
        result_df = pd.DataFrame(columns=['testClassName','testMethodName','potentialTargetQualifiedName','AAA','Name Similarity','Assert Distance','Tag-Mock','Tag-New','Tag-Test','Tag-Get','Tag-Set'])

        for project_path_list in projects:
            project_df = self.one_project_tfidf(project_path_list)
            result_df = pd.concat([result_df, project_df], ignore_index=True)
        print('TFIDF dataframes merged!')
        # build assert distance
        print('Building assert distance & Tag info...')
        for i in range(result_df.shape[0]):
            if self.assertDistanceMethod == 'basic':
                assert_dis = 0
                casename = result_df['testMethodName'][i]
                while assert_dis < (result_df.shape[0]-i):
                    if casename != result_df['testMethodName'][(i+assert_dis)]:
                        assert_dis = -1 #TODO:
                        break
                    if re.search(r'ASSERT',result_df['potentialTargetQualifiedName'][(i+assert_dis)]) :
                        break
                    else:
                        assert_dis += 1

                result_df.loc[i,'Assert Distance'] = assert_dis
        
            # build Tag features
            if self.tag:
                try: 
                    tag_keyword = result_df['potentialTargetQualifiedName'][i].lstrip().split(' ')[0]
                except Exception as e:
                    print(e)
                    print('{0} line has no 0 position'.format(i))
                if tag_keyword == 'MOCK':
                    # print("Mock")
                    # print(tag_keyword)
                    result_df.loc[i, 'Tag-Mock'] = 1
                elif tag_keyword == 'NEW':
                    # print("New")
                    # print(tag_keyword)
                    result_df.loc[i, 'Tag-New'] = 1
                elif tag_keyword == 'TEST':
                    # print("Test")
                    # print(tag_keyword)
                    result_df.loc[i, 'Tag-Test'] = 1

                pattern_1 = re.compile('.get')
                pattern_2 = re.compile('.set')
                str= result_df['potentialTargetQualifiedName'][i].lstrip()
                m_1 = pattern_1.search(str, re.I)
                m_2 = pattern_2.search(str, re.I)
                if m_1 != None:
                    result_df.loc[i, 'Tag-Get'] = 1
                elif m_2 != None:
                    result_df.loc[i, 'Tag-Set'] =1
        print('ALL the building finished!')
        return result_df

if __name__ == "__main__":
    # local test
    data = Data('./all Projects/Data_NoLevel/','tfidf','basic',tag=True,data_structure='clean').df
    print(data.head())
    data.to_csv('./tfidf_df.csv', index=False)
