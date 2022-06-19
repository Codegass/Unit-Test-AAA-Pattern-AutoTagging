# %%
import os, re
from matplotlib.pyplot import title
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
from DataClean import Data
import plotly.express as px
import plotly.graph_objects as go
import statistics

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


def AAA_analysis(pre_fix_path, project):
    data = Data(pre_fix_path+project+'/','edit distance', 'basic', tag=True, data_structure='clean').df
    groups = data.groupby('testMethodName')
    AR_case_level_count_list = []
    AS_case_level_count_list = []
    AC_case_level_count_list = []
    ASSERT_DENSITY_list = []
    overall_statement_count_list = []
    for group_name, group in groups:
        count_series = group["AAA"].value_counts()
        AR_count, AS_count, AC_count = 0, 0, 0
        for i, j in count_series.items():
            if i == 0:
                # AR_case_level_count_list.append(j)
                AR_count = j
            elif i == 2:
                # AS_case_level_count_list.append(j)
                AS_count = j
            elif i == 1:
                # AC_case_level_count_list.append(j)
                AC_count = j
        AR_case_level_count_list.append(AR_count)
        AS_case_level_count_list.append(AS_count)
        AC_case_level_count_list.append(AC_count)
        ASSERT_DENSITY_list.append(AS_count/(AR_count+AS_count+AC_count))
        overall_statement_count_list.append(AR_count+AS_count+AC_count)

        if AC_count == 0:
            print("AC is 0")
            print(group['testClassName'].iloc[0]+'.'+group_name)
        if AS_count == 0:
            print("AS is 0")
            print(group['testClassName'].iloc[0]+'.'+group_name)
        if AS_count >100:
            print(f"AS is too much {AS_count}")
            print(group['testClassName'].iloc[0]+'.'+group_name)
        if AR_count >100:
            print(f"AR is too much {AR_count}")
            print(group['testClassName'].iloc[0]+'.'+group_name)
        if AC_count >10:
            print(f"AC is too much {AC_count}")
            print(group['testClassName'].iloc[0]+'.'+group_name)
        
    print('=====================')
    print(project)
    print('total metrics')
    print(data['AAA'].value_counts())
    print('MAX')
    print('AR: {}'.format(max(AR_case_level_count_list)))
    print('AS: {}'.format(max(AS_case_level_count_list)))
    print('AC: {}'.format(max(AC_case_level_count_list)))
    print('ASSERT DENSITY: {}'.format(max(ASSERT_DENSITY_list)))
    print('overall statement: {}'.format(max(overall_statement_count_list)))
    print('MIN')
    print('AR: {}'.format(min(AR_case_level_count_list)))
    print('AS: {}'.format(min(AS_case_level_count_list)))
    print('AC: {}'.format(min(AC_case_level_count_list)))
    print('ASSERT DENSITY: {}'.format(min(ASSERT_DENSITY_list)))
    print('overall statement: {}'.format(min(overall_statement_count_list)))
    print('MEAN')
    print('AR: {}'.format(np.mean(AR_case_level_count_list)))
    print('AS: {}'.format(np.mean(AS_case_level_count_list)))
    print('AC: {}'.format(np.mean(AC_case_level_count_list)))
    print('ASSERT DENSITY: {}'.format(np.mean(ASSERT_DENSITY_list)))
    print('overall statement: {}'.format(np.mean(overall_statement_count_list)))
    print('MODE')
    print('AR: {}'.format(statistics.mode(AR_case_level_count_list)))
    print('AS: {}'.format(statistics.mode(AS_case_level_count_list)))
    print('AC: {}'.format(statistics.mode(AC_case_level_count_list)))
    print('ASSERT DENSITY: {}'.format(statistics.mode(ASSERT_DENSITY_list)))
    print('overall statement: {}'.format(statistics.mode(overall_statement_count_list)))
    print('MEDIAN')
    print('AR: {}'.format(statistics.median(AR_case_level_count_list)))
    print('AS: {}'.format(statistics.median(AS_case_level_count_list)))
    print('AC: {}'.format(statistics.median(AC_case_level_count_list)))
    print('ASSERT DENSITY: {}'.format(statistics.median(ASSERT_DENSITY_list)))
    print('overall statement: {}'.format(statistics.median(overall_statement_count_list)))
    print('=====================')
    return AC_case_level_count_list, AR_case_level_count_list, AS_case_level_count_list, ASSERT_DENSITY_list


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
    
# %%
if True:
#### Each A number in Each Project
    project_list = ['accumulo','cloudstack','druid','dubbo']
    pre_fix_path = './all Projects/Data_NoLevel/'
    print('=== === === === ===')
    for project in project_list:
        # exec(f'data_{project} = Data(pre_fix_path+project+\'/\',\'edit distance\', \'basic\', tag=True, data_structure=\'clean\').df')
        # print(project)
        # exec(f'print(data_{project}[\'AAA\'].value_counts())')
        exec(f'{project}_AC_distribution,{project}_AR_distribution,{project}_AS_distribution,{project}_assert_density_distribution = AAA_analysis(pre_fix_path,project)')
# %%
if True:
    ### generate the histogram of AC ###
    ac_fig = go.Figure(layout_title_text='Action distribution of each project')
    for project in project_list:
        exec(f'ac_fig.add_trace(go.Histogram(x={project}_AC_distribution,name=\'{project}\',nbinsx=30))')
    
    # Overlay both histograms
    # ac_fig.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # ac_fig.update_traces(opacity=0.4)
    ac_fig.update_layout(barmode='stack')
    ac_fig.update_traces(xbins_size=1)
    ac_fig.show()

    ####################################
    ### generate the histogram of AR ###
    ar_fig = go.Figure(layout_title_text='Arrangement distribution of each project')
    for project in project_list:
        exec(f'ar_fig.add_trace(go.Histogram(x={project}_AR_distribution,name=\'{project}\',nbinsx=30))')

    # Overlay both histograms
    # ar_fig.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # ar_fig.update_traces(opacity=0.4)
    ar_fig.update_layout(barmode='stack')
    ar_fig.update_traces(xbins_size=1)
    ar_fig.show()

    ####################################
    ### generate the histogram of AS ###
    as_fig = go.Figure(layout_title_text='Assertion distribution of each project')
    for project in project_list:
        exec(f'as_fig.add_trace(go.Histogram(x={project}_AS_distribution,name=\'{project}\',nbinsx=30))')

    # Overlay both histograms
    # as_fig.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # as_fig.update_traces(opacity=0.4)
    as_fig.update_layout(barmode='stack')
    as_fig.update_traces(xbins_size=1)
    as_fig.show()

    ####################################
    ### generate the histogram of assert density ###
    assert_density_fig = go.Figure(layout_title_text='Assert density distribution of each project')
    for project in project_list:
        exec(f'assert_density_fig.add_trace(go.Histogram(x={project}_assert_density_distribution,name=\'{project}\',nbinsx=30))')

    # Overlay both histograms
    # assert_density_fig.update_layout(barmode='overlay')
    # # Reduce opacity to see both histograms
    # assert_density_fig.update_traces(opacity=0.4)
    assert_density_fig.update_layout(barmode='stack')
    # assert_density_fig.update_traces(xbins_size=2)
    assert_density_fig.show()

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
# %%
if True:
    from plotly.subplots import make_subplots

    ### generate the histogram of AC ###
    ac_fig = make_subplots(rows=2, cols=2, start_cell="top-left",x_title='Action Quantity in Each Case',y_title='Case Quantity')
    for project in range(len(project_list)):
        if project == 0:
            row = 1
            col = 1
            project_name = 'accumulo'
        elif project == 1:
            row = 1
            col = 2
            project_name = 'cloudstack'
        elif project == 2:
            row = 2
            col = 1
            project_name = 'druid'
        elif project == 3:
            row = 2
            col = 2
            project_name = 'dubbo'
            
        exec(f'ac_fig.add_trace(go.Histogram(x={project_name}_AC_distribution,name=\'{project_name}\',nbinsx=20),row={row},col={col})')
    
    ac_fig.show()

    ####################################
    ### generate the histogram of AR ###
    ar_fig = make_subplots(rows=2, cols=2, start_cell="top-left",x_title='Arrangement Quantity in Each Case',y_title='Case Quantity')
    for project in range(len(project_list)):
        if project == 0:
            row = 1
            col = 1
            project_name = 'accumulo'
        elif project == 1:
            row = 1
            col = 2
            project_name = 'cloudstack'
        elif project == 2:
            row = 2
            col = 1
            project_name = 'druid'
        elif project == 3:
            row = 2
            col = 2
            project_name = 'dubbo'
            
        exec(f'ar_fig.add_trace(go.Histogram(x={project_name}_AR_distribution,name=\'{project_name}\',nbinsx=20),row={row},col={col})')
    
    ar_fig.show()

    ####################################
    ### generate the histogram of AS ###
    as_fig = make_subplots(rows=2, cols=2, start_cell="top-left",x_title='Assertion Quantity in Each Case',y_title='Case Quantity')
    for project in range(len(project_list)):
        if project == 0:
            row = 1
            col = 1
            project_name = 'accumulo'
        elif project == 1:
            row = 1
            col = 2
            project_name = 'cloudstack'
        elif project == 2:
            row = 2
            col = 1
            project_name = 'druid'
        elif project == 3:
            row = 2
            col = 2
            project_name = 'dubbo'
            
        exec(f'as_fig.add_trace(go.Histogram(x={project_name}_AS_distribution,name=\'{project_name}\',nbinsx=20),row={row},col={col})')
    
    as_fig.show()

    ####################################
    ### generate the histogram of assert density ###
    assert_density_fig = make_subplots(rows=2, cols=2, start_cell="top-left",x_title='Assert Density in Each Case',y_title='Case Quantity')
    for project in range(len(project_list)):
        if project == 0:
            row = 1
            col = 1
            project_name = 'accumulo'
        elif project == 1:
            row = 1
            col = 2
            project_name = 'cloudstack'
        elif project == 2:
            row = 2
            col = 1
            project_name = 'druid'
        elif project == 3:
            row = 2
            col = 2
            project_name = 'dubbo'
            
        exec(f'assert_density_fig.add_trace(go.Histogram(x={project_name}_assert_density_distribution,name=\'{project_name}\',nbinsx=20),row={row},col={col})')

    assert_density_fig.show()
# %%



if True:
    def specific_assert(pre_fix_path, project):
        ''' split the assert into direct assert and help assert'''
        data = Data(pre_fix_path+project+'/','edit distance', 'basic', tag=True, data_structure='clean').df
        groups = data.groupby('testMethodName')
        direct_assert_list = []
        help_assert_list = []
        direct_assert_density_list = []
        help_assert_density_list = []

        for group_name, group in groups:
            direct_assert_count = 0
            help_assert_count = 0
            # direct_assert_density = 0
            # help_assert_density = 0

            for index in range(group['AAA'].shape[0]):
                if group['AAA'].iloc[index] == 2:
                    ### Judge the assert type ###
                    if re.search('ASSERT', group['potentialTargetQualifiedName'].iloc[index],re.IGNORECASE):
                        direct_assert_count += 1
                    elif re.search('verify', group['potentialTargetQualifiedName'].iloc[index].split('.')[-1],re.IGNORECASE):
                        direct_assert_count += 1
                    elif re.search('check', group['potentialTargetQualifiedName'].iloc[index].split('.')[-1],re.IGNORECASE):
                        direct_assert_count += 1
                    else:
                        help_assert_count += 1
            direct_assert_list.append(direct_assert_count)
            help_assert_list.append(help_assert_count)

            direct_assert_density_list.append(direct_assert_count/group.shape[0])
            help_assert_density_list.append(help_assert_count/group.shape[0])

            if direct_assert_count == 0:
                print("AS")
                print(group['testClassName'].iloc[0]+'.'+group_name)

        print(project)
        print('total Direct Assert:', sum(direct_assert_list))
        print('total Help Assert:', sum(help_assert_list))
        print('MAX')
        print('Direct Assert: {}'.format(max(direct_assert_list)))
        print('Help Assert: {}'.format(max(help_assert_list)))
        print('Direct Assert Density: {}'.format(max(direct_assert_density_list)))
        print('Help Assert Density: {}'.format(max(help_assert_density_list)))
        print('MIN')
        print('Direct Assert: {}'.format(min(direct_assert_list)))
        print('Help Assert: {}'.format(min(help_assert_list)))
        print('Direct Assert Density: {}'.format(min(direct_assert_density_list)))
        print('Help Assert Density: {}'.format(min(help_assert_density_list)))
        print('MEAN')
        print('Direct Assert: {}'.format(np.mean(direct_assert_list)))
        print('Help Assert: {}'.format(np.mean(help_assert_list)))
        print('Direct Assert Density: {}'.format(np.mean(direct_assert_density_list)))
        print('Help Assert Density: {}'.format(np.mean(help_assert_density_list)))
        print('MODE')
        print('Direct Assert: {}'.format(statistics.mode(direct_assert_list)))
        print('Help Assert: {}'.format(statistics.mode(help_assert_list)))
        print('Direct Assert Density: {}'.format(statistics.mode(direct_assert_density_list)))
        print('Help Assert Density: {}'.format(statistics.mode(help_assert_density_list)))
        print('MEDIAN')
        print('Direct Assert: {}'.format(statistics.median(direct_assert_list)))
        print('Help Assert: {}'.format(statistics.median(help_assert_list)))
        print('Direct Assert Density: {}'.format(statistics.median(direct_assert_density_list)))
        print('Help Assert Density: {}'.format(statistics.median(help_assert_density_list)))

        return direct_assert_list, help_assert_list, direct_assert_density_list, help_assert_density_list

    project_list = ['accumulo','cloudstack','druid','dubbo']
    pre_fix_path = './all Projects/Data_NoLevel/'
    print('=== === === === ===')
    overall_direct_assert_density_list = []
    overall_help_assert_list = []
    overall_direct_assert_list = []
    for project in project_list:
        exec(f'{project}_direct_assert_list, {project}_help_assert_list, {project}_direct_assert_density_list, {project}_help_assert_density_list = specific_assert(pre_fix_path,project)')
        exec(f'overall_direct_assert_density_list += {project}_direct_assert_density_list')
        exec(f'overall_help_assert_list += {project}_help_assert_list')
        exec(f'overall_direct_assert_list += {project}_direct_assert_list')

    print('=== === === === ===')
    print('total Direct Assert:', sum(overall_direct_assert_list))
    print('total Help Assert:', sum(overall_help_assert_list))
    print(f"Direct Assert Density: MAX {max(overall_direct_assert_density_list)} Min {min(overall_direct_assert_density_list)} MEAN {np.mean(overall_direct_assert_density_list)} MODE {statistics.mode(overall_direct_assert_density_list)} MEDIAN {statistics.median(overall_direct_assert_density_list)}")
    print(f'Direct Assert: MAX {max(overall_direct_assert_list)} Min {min(overall_direct_assert_list)} MEAN {np.mean(overall_direct_assert_list)} MODE {statistics.mode(overall_direct_assert_list)} MEDIAN {statistics.median(overall_direct_assert_list)}')
    print(f'Help Assert: MAX {max(overall_help_assert_list)} Min {min(overall_help_assert_list)} MEAN {np.mean(overall_help_assert_list)} MODE {statistics.mode(overall_help_assert_list)} MEDIAN {statistics.median(overall_help_assert_list)}')

# %%
if True:
    ### generate the histogram of direct AS ###
    direct_fig = go.Figure(layout_title_text='Direct Assertion distribution of each project')
    for project in project_list:
        exec(f'direct_fig.add_trace(go.Histogram(x={project}_direct_assert_list,name=\'{project}\',nbinsx=30))')
    
    direct_fig.update_layout(barmode='stack')
    direct_fig.update_traces(xbins_size=1)
    direct_fig.show()

    ### generate the histogram of help AS ###
    help_fig = go.Figure(layout_title_text='Help Assertion distribution of each project')
    for project in project_list:
        exec(f'help_fig.add_trace(go.Histogram(x={project}_help_assert_list,name=\'{project}\',nbinsx=30))')
    
    help_fig.update_layout(barmode='stack')
    help_fig.update_traces(xbins_size=1)
    help_fig.show()

    ### generate the histogram of direct AS density ###
    direct_density_fig = go.Figure(layout_title_text='Direct Assertion density distribution of each project')
    for project in project_list:
        exec(f'direct_density_fig.add_trace(go.Histogram(x={project}_direct_assert_density_list,name=\'{project}\'))')
    
    direct_density_fig.update_layout(barmode='stack')
    # direct_density_fig.update_traces(xbins_size=2)
    direct_density_fig.show()

    ### generate the histogram of help AS density ###
    help_density_fig = go.Figure(layout_title_text='Help Assertion density distribution of each project')
    for project in project_list:
        exec(f'help_density_fig.add_trace(go.Histogram(x={project}_help_assert_density_list,name=\'{project}\'))')
    
    help_density_fig.update_layout(barmode='stack')
    # help_density_fig.update_traces(xbins_size=2)
    help_density_fig.show()
# %%
if True:
    import plotly.figure_factory as ff

    group_labels = ['accumulo','cloudstack','druid','dubbo']
    colors = ['#333F44', '#37AA9C', '#94F3E4', '#FFC300']

    ## Action
    hist_data = [accumulo_AC_distribution,cloudstack_AC_distribution, druid_AC_distribution, dubbo_AC_distribution]
    # Create distplot with curve_type set to 'normal'
    ac_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)

    # Add title
    ac_fig.update_layout(title_text='Action distribution of each project')
    ac_fig.show()

    ## Assertion
    hist_data = [accumulo_AS_distribution,cloudstack_AS_distribution, druid_AS_distribution, dubbo_AS_distribution]
    as_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
    as_fig.update_layout(title_text='Assertion distribution of each project')
    as_fig.show()

    ## Arrangement
    hist_data = [accumulo_AR_distribution,cloudstack_AR_distribution, druid_AR_distribution, dubbo_AR_distribution]
    ar_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
    ar_fig.update_layout(title_text='Arrangement distribution of each project')
    ar_fig.show()

    ## Assertion Density
    hist_data = [accumulo_assert_density_distribution,cloudstack_assert_density_distribution, druid_assert_density_distribution, dubbo_assert_density_distribution]
    density_fig = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
    density_fig.update_layout(title_text='Assertion density distribution of each project')
    density_fig.show()

# %%
pd.DataFrame({'accumulo':accumulo_AC_distribution,'cloudstack':cloudstack_AC_distribution,'druid':druid_AC_distribution,'dubbo':dubbo_AC_distribution}).to_csv('./all Projects/Data_NoLevel/temp/AC_distribution.csv')
# %%
pd.concat([pd.DataFrame(accumulo_AC_distribution),pd.DataFrame(cloudstack_AC_distribution),pd.DataFrame(druid_AC_distribution),pd.DataFrame(dubbo_AC_distribution)],axis=1).to_csv('./all Projects/temp/AC_distribution.csv')
# %%
pd.concat([pd.DataFrame(accumulo_AS_distribution),pd.DataFrame(cloudstack_AS_distribution),pd.DataFrame(druid_AS_distribution),pd.DataFrame(dubbo_AS_distribution)],axis=1).to_csv('./all Projects/temp/AS_distribution.csv')
pd.concat([pd.DataFrame(accumulo_AR_distribution),pd.DataFrame(cloudstack_AR_distribution),pd.DataFrame(druid_AR_distribution),pd.DataFrame(dubbo_AR_distribution)],axis=1).to_csv('./all Projects/temp/AR_distribution.csv')
pd.concat([pd.DataFrame(accumulo_assert_density_distribution),pd.DataFrame(cloudstack_assert_density_distribution),pd.DataFrame(druid_assert_density_distribution),pd.DataFrame(dubbo_assert_density_distribution)],axis=1).to_csv('./all Projects/temp/assert_density_distribution.csv')
# %%
pd.concat([pd.DataFrame(accumulo_direct_assert_list),pd.DataFrame(cloudstack_direct_assert_list),pd.DataFrame(druid_direct_assert_list),pd.DataFrame(dubbo_direct_assert_list)],axis=1).to_csv('./all Projects/temp/direct_assert_list.csv')
pd.concat([pd.DataFrame(accumulo_help_assert_list),pd.DataFrame(cloudstack_help_assert_list),pd.DataFrame(druid_help_assert_list),pd.DataFrame(dubbo_help_assert_list)],axis=1).to_csv('./all Projects/temp/help_assert_list.csv')
pd.concat([pd.DataFrame(accumulo_direct_assert_density_list),pd.DataFrame(cloudstack_direct_assert_density_list),pd.DataFrame(druid_direct_assert_density_list),pd.DataFrame(dubbo_direct_assert_density_list)],axis=1).to_csv('./all Projects/temp/direct_assert_density_list.csv')
pd.concat([pd.DataFrame(accumulo_help_assert_density_list),pd.DataFrame(cloudstack_help_assert_density_list),pd.DataFrame(druid_help_assert_density_list),pd.DataFrame(dubbo_help_assert_density_list)],axis=1).to_csv('./all Projects/temp/help_assert_density_list.csv')
# %%
pd.DataFrame(overall_direct_assert_density_list).to_csv('./all Projects/temp/overall_direct_assert_density_list.csv')

# %%
## Count AAA
