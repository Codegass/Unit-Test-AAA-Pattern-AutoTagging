import os, re
from numpy.lib.function_base import percentile
import pandas as pd
from DataClean import Data

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def readFileCountAtTest(path):
    count = 0
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            if re.search('@Test',line):
                count += 1
        return count

def countFiles(path):
    count = 0
    for file in findAllFile(path):
        count +=1
    return count

def countTestFileAndCases(path):
    '''匹配 IT/Test 结尾， @test case'''
    test_files_number = 0
    test_cases_number = 0

    for file in findAllFile(path):
        if file == (path + '.DS_Store'):
            continue
        # count file
        try:
            path_list = file.split('/')
            if re.search('Test',path_list[-1]) or re.search('IT', path_list[-1]):
                test_files_number += 1
                test_cases_number = test_cases_number + readFileCountAtTest(file)
        except Exception as e:
            # print(file)
            # print(e)
            pass

    # print('test files: {}'.format(test_files_number))
    # print('test cases {}'.format(test_cases_number))
    return (test_files_number,test_cases_number)
        
def findhemcrest(path):
    ''' 找到所有用了 hemcrest 的 file '''
    result_list = []
    for file in findAllFile(path):
        if file == (path + '.DS_Store'):
            continue
        if re.search(r'\.class', file) or re.search(r'\.orc', file):
            # print('class happened: {}'.format(file))
            continue
        try:
            path_list = file.split('/')
            if re.search('Test',path_list[-1]) or re.search('IT', path_list[-1]):
                with open(file) as f:
                    lines = f.readlines()
                    for line in lines:
                        if re.search('hamcrest',line):
                            result_list.append(file.split('/')[-1].split('.')[0])
                            # print(file)
                            break
                        # if re.search(r'assertThat\(',line):
                        #     print(file)      
        except Exception as e:
            print(file)
            print(e)
            # pass
    return result_list

def find_verify_and_check_statement(path):
    result_list = []
    pass
        


if __name__ == '__main__':
    print('start to count...')
    path_prefix = '/Users/chenhao/Documents/Code/Java/' #if the project location changed, please change this prefix
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

    #Accomulo cases
    accomulo_file_number = 0
    accimulo_case_number = 0
    for path in accomulo_project_path_list:
        # abs_path = path_prefix + path
        abs_path = path
        test_files_number,test_cases_number = countTestFileAndCases(abs_path)
        accomulo_file_number = accomulo_file_number + test_files_number
        accimulo_case_number = accimulo_case_number + test_cases_number
    print('accomulo files count: {}'.format(accomulo_file_number))
    print('accomulo cases count: {}'.format(accimulo_case_number))
    print('=== === === === ===')

    druid_file_number = 0
    druid_case_number = 0
    for path in druid_project_path_list:
        abs_path = path
        test_files_number,test_cases_number = countTestFileAndCases(abs_path)
        druid_file_number = druid_file_number + test_files_number
        druid_case_number = druid_case_number + test_cases_number

    print('druid files count: {}'.format(druid_file_number))
    print('druid cases count: {}'.format(druid_case_number))
    print('=== === === === ===')

    cloudstack_file_number = 0
    cloudstack_case_number = 0
    for path in cloudstack_project_path_list:
        abs_path = path
        test_files_number,test_cases_number = countTestFileAndCases(abs_path)
        cloudstack_file_number = cloudstack_file_number + test_files_number
        cloudstack_case_number = cloudstack_case_number + test_cases_number

    print('cloudstack files count: {}'.format(cloudstack_file_number))
    print('cloudstack cases count: {}'.format(cloudstack_case_number))
    print('=== === === === ===')

    dubbo_file_number = 0
    dubbo_case_number = 0
    for path in dubbo_project_path_list:
        abs_path = path
        test_files_number,test_cases_number = countTestFileAndCases(abs_path)
        dubbo_file_number = dubbo_file_number + test_files_number
        dubbo_case_number = dubbo_case_number + test_cases_number

    print('=== === === === ===')
    print('dubbo files count: {}'.format(countFiles('/Users/chenhao/Documents/Code/Java/dubbo-dubbo-2.7.7/')))
    print('dubbo test files count: {}'.format(dubbo_file_number))
    print('dubbo cases count: {}'.format(dubbo_case_number))
    print('=== === === === ===')

    print('accomulo files count: {}'.format(countFiles('/Users/chenhao/Documents/Code/Java/accumulo-rel-2.0.0/')))
    print('accomulo test files count: {}'.format(accomulo_file_number))
    print('accomulo cases count: {}'.format(accimulo_case_number))
    print('=== === === === ===')

    print('cloudstack files count: {}'.format(countFiles('/Users/chenhao/Documents/Code/Java/cloudstack-4.13.1.0/')))
    print('cloudstack test files count: {}'.format(cloudstack_file_number))
    print('cloudstack cases count: {}'.format(cloudstack_case_number))
    print('=== === === === ===')

    print('druid files count: {}'.format(countFiles('/Users/chenhao/Documents/Code/Java/druid-druid-0.19.0/')))
    print('druid test files count: {}'.format(druid_file_number))
    print('druid cases count: {}'.format(druid_case_number))
    print('=== === === === ===')

#### 鉴别 hamcrest 对 assert 的影响

    All = accomulo_project_path_list+dubbo_project_path_list+druid_project_path_list+cloudstack_project_path_list
    hamcrestlist =[]
    for path in All:
        hamcrest = findhemcrest(path)
        if hamcrest:
            hamcrestlist = hamcrestlist +hamcrest # hamcrestlist is the list contains all the file which use the hamcrest package.
    print(hamcrestlist)

    for caseName in findAllFile('./all Projects/'):
        if re.search('xlsx',caseName):
            for filename_ in hamcrestlist:
                if re.search(filename_,caseName):
                    # print('check {}'.format(caseName))
                    df = pd.read_excel(caseName)
                    for item in df['Naming Convention is meanningful'][3:]:
                        if re.search('hamcrest',item):
                            print(caseName)
                        elif re.search('assertThat',item):
                            print(caseName)
                            break
                    break
     
###### H1: all the files with verify or check
    casenumber = 0
    statementnumber_all = 0
    for caseName in findAllFile('./all Projects/Data_NoLevel/'):
    # for caseName in findAllFile('./all Projects/P3-W/'):
        if re.search('csv', caseName):
            statementnumber = 0
            df = pd.read_csv(caseName)
            # print(df['Naming Convention is meanningful'][3:].size)
            for index in range(df['potentialTargetQualifiedName'].size):
                if re.search('verify',df['potentialTargetQualifiedName'][index].split('.')[-1],re.IGNORECASE) and df['AAA'][index] == 2:
                    statementnumber += 1
                    # print(df['Naming Convention is meanningful'][index+3].split('.')[-1])
                elif re.search('check', df['potentialTargetQualifiedName'][index].split('.')[-1],re.IGNORECASE) and df['AAA'][index] == 2:
                    statementnumber += 1
                
            if statementnumber != 0:
                casenumber += 1
                
                statementnumber_all += statementnumber

    print('=== === === === ===')
    print('H1 Verify and Check')
    c_percentile = casenumber/500
    print(f'{casenumber} cases({c_percentile}) has these unsual assert')
    s_percentile = statementnumber_all/19352
    print(f'{statementnumber_all} statements({s_percentile}) are these kind of assert')
    a_persentile = statementnumber_all/5940
    print(f'{statementnumber_all} asserts({a_persentile} are these kind of asserts in overall asserts )')

###### H5: assert before action
    #TODO:Need to check the assert is that all the 
    casenumber = 0
    statementnumber_all = 0
    for caseName in findAllFile('./all Projects/Data_NoLevel/'):
    # for caseName in findAllFile('./all Projects/P3-W/'):
        if re.search('csv', caseName):
            statementnumber = 0
            df = pd.read_csv(caseName)
            
            for index in range(df['potentialTargetQualifiedName'].size):
                if re.search('verify',df['potentialTargetQualifiedName'][index].split('.')[-1],re.IGNORECASE) and df['AAA'][index] == 0:
                    statementnumber += 1
                elif re.search('check', df['potentialTargetQualifiedName'][index].split('.')[-1],re.IGNORECASE) and df['AAA'][index] == 0:
                    statementnumber += 1
                elif re.search('assert',df['potentialTargetQualifiedName'][index].split('.')[-1],re.IGNORECASE) and df['AAA'][index] == 0:
                    statementnumber += 1

            if statementnumber != 0:
                casenumber += 1
                # print(caseName)
                statementnumber_all += statementnumber
    print('=== === === === ===')
    print('H5 Assert before action')
    c_percentile = casenumber/500
    print(f'{casenumber} cases({c_percentile}) has these assert before action')
    s_percentile = statementnumber_all/19352
    print(f'{statementnumber_all} statements({s_percentile}) are these kind of assert')
    

#### Each A number in Each Project
    project_list = ['accumulo','cloudstack','druid','dubbo']
    pre_fix_path = './all Projects/Data_NoLevel/'
    print('=== === === === ===')
    for project in project_list:
        exec(f'data_{project} = Data(pre_fix_path+project+\'/\',\'edit distance\', \'basic\', tag=True, data_structure=\'clean\').df')
        print(project)
        exec(f'print(data_{project}[\'AAA\'].value_counts())')

#### count H3 setter getter 
    # casenumber = 0
    # statementnumber_all = 0
    # for caseName in findAllFile('./all Projects/Data_NoLevel/'):
    # # for caseName in findAllFile('./all Projects/P3-W/'):
    #     if re.search('csv', caseName):
    #         statementnumber = 0
    #         df = pd.read_csv(caseName)
    #         # print(df['Naming Convention is meanningful'][3:].size)
    #         for index in range(df['potentialTargetQualifiedName'].size):
    #             if re.search('GET',df['potentialTargetQualifiedName'][index],re.IGNORECASE) :
    #                 statementnumber += 1
    #                 # print(df['Naming Convention is meanningful'][index+3].split('.')[-1])
    #             elif re.search('SET', df['potentialTargetQualifiedName'][index],re.IGNORECASE) :
    #                 statementnumber += 1
    #             elif re.search('MOCK', df['potentialTargetQualifiedName'][index],re.IGNORECASE) :
    #                 statementnumber += 1
    #             elif re.search('NEW', df['potentialTargetQualifiedName'][index],re.IGNORECASE) :
    #                 statementnumber += 1
                
    #         if statementnumber != 0:
    #             casenumber += 1
                
    #             statementnumber_all += statementnumber

    # print('=== === === === ===')
    # print('H1 Verify and Check')
    # c_percentile = casenumber/500
    # print(f'{casenumber} cases({c_percentile}) has these unsual assert')
    # s_percentile = statementnumber_all/19352
    # print(f'{statementnumber_all} statements({s_percentile}) are these kind of assert')
    # a_persentile = statementnumber_all
    # print(f'{statementnumber_all} asserts({a_persentile} are these kind of asserts in overall asserts )')