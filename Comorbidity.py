import pandas as pd
import numpy as np
import datetime
import os
import yaml
import xlrd
from shutil import copyfile
from config import category,health_count,disease_count,diseasename,diseaseicd,ORthrehold, sup_filter
from scipy.stats import nchypergeom_fisher
import scipy.stats as stats
import math
def get_df_outer(df_outer, df1_sup_sum, df2_sup_sum, df1_pNum, df2_pNum):
    # pseudo count
    df_outer['#SUP_x_2'] = df_outer['#SUP_x']
    df_outer['#SUP_x_2'] = df_outer['#SUP_x_2'].fillna(value=0)
    for index, row in df_outer['#SUP_x_2'].items():
        df_outer['#SUP_x_2'][index] += 0.5
    df1_sup_sum = df_outer['#SUP_x_2'].sum()
    df_outer['#SUP_y_2'] = df_outer['#SUP_y']
    df_outer['#SUP_y_2'] = df_outer['#SUP_y_2'].fillna(value=0)
    for index, row in df_outer['#SUP_y_2'].items():
        df_outer['#SUP_y_2'][index] += 0.5
    df2_sup_sum = df_outer['#SUP_y_2'].sum()
    df1_pNum += 1
    df2_pNum += 1

    # PJ
    df_outer['XPJ_2'] = df_outer['#SUP_x_2'] / df1_sup_sum
    df_outer['YPJ_2'] = df_outer['#SUP_y_2']/ df2_sup_sum
    df_outer['PJ_2'] = (df_outer['XPJ_2'] + df_outer['YPJ_2']) / 2
    df_outer['PJT_2'] = (df_outer['#SUP_x_2'] / df1_sup_sum + df_outer['#SUP_y_2'] / df2_sup_sum) / 2
    # OR
    df_outer['OR_x_2'] = df_outer['#SUP_x_2'] / (df1_pNum - df_outer['#SUP_x_2'])
    df_outer['OR_y_2'] = df_outer['#SUP_y_2'] / (df2_pNum - df_outer['#SUP_y_2'])
    df_outer['OR_2'] = df_outer['OR_x_2'] / df_outer['OR_y_2']
   
    return df_outer



def get_df_outer_2(df_outer, df1_sup_sum, df2_sup_sum, df1_pNum, df2_pNum):
    # PJ
    df_outer['XPJ'] = np.where((pd.isnull(df_outer['#SUP_x'])),
                               np.nan,
                               df_outer['#SUP_x'] / df1_sup_sum)
    df_outer['YPJ'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                               np.nan,
                               df_outer['#SUP_y'] / df2_sup_sum)
    df_outer['PJ'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                              df_outer['XPJ'],
                              (df_outer['XPJ'] + df_outer['YPJ']) / 2)
    df_outer['PJT'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                               df_outer['#SUP_x'] / df1_sup_sum,
                               (df_outer['#SUP_x'] / df1_sup_sum + df_outer['#SUP_y'] / df2_sup_sum) / 2)
    # OR
    df_outer['OR_x'] = np.where((pd.isnull(df_outer['#SUP_x'])),
                                np.nan,
                                df_outer['#SUP_x'] / (df1_pNum - df_outer['#SUP_x']))
    df_outer['OR_y'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                                np.nan,
                                df_outer['#SUP_y'] / (df2_pNum - df_outer['#SUP_y']))
    df_outer['OR'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                              float("inf"),
                              df_outer['OR_x'] / df_outer['OR_y'])
     #test P
    '''df_outer['P_x'] =  np.where((pd.isnull(df_outer['#SUP_x'])),
                                np.nan,
                                hypergeom.pmf(df_outer['#SUP_x'], df_outer['#SUP_x']+df_outer['No#SUP_x']+df_outer['#SUP_y']+df_outer['No#SUP_y'], df_outer['#SUP_x']+df_outer['No#SUP_x'], df_outer['#SUP_x']+df_outer['#SUP_y'], 0))
    df_outer['P_y'] =  np.where((pd.isnull(df_outer['#SUP_x'])),
                                np.nan,
                                hypergeom.pmf(0, df_outer['#SUP_x']+df_outer['No#SUP_x']+df_outer['#SUP_y']+df_outer['No#SUP_y'], df_outer['#SUP_x']+df_outer['No#SUP_x'], df_outer['#SUP_y'], 0))
    df_outer['Pmf_1'] = (df_outer['P_x']/(1-df_outer['P_x'])) /(df_outer['P_y']/(1-df_outer['P_y']))
    df_outer['Pmf_1'].apply(lambda x: float(x))
    df_outer['Pmf_2'] = np.log(df_outer['Pmf_1'])'''
    Y = df_outer['#SUP_y'].fillna(value=0)
    NoY = df_outer['No#SUP_y'].fillna(value=df2_pNum)
    X = df_outer['#SUP_x'].fillna(value=0)
    NoX = df_outer['No#SUP_x'].fillna(value=df1_pNum)
    #X , N, ma, mb = X, df_outer['#SUP_x']+df_outer['No#SUP_x']+df_outer['#SUP_y']+df_outer['No#SUP_y'], df_outer['#SUP_x']+df_outer['No#SUP_x'], df_outer['#SUP_x']+df_outer['#SUP_y']
    x , N, ma, mb = X, X+NoX+Y+NoY, X+NoX, X+Y
    odds = math.exp(0)#0.1
    df_outer["p1"] = 1 - nchypergeom_fisher.cdf(x-1, N, ma, mb, odds)
    df_outer["p2"] = nchypergeom_fisher.cdf(x, N, ma, mb, odds)
    df_outer["a1"] = df_outer["p1"] + nchypergeom_fisher.cdf(nchypergeom_fisher.ppf(df_outer["p1"], N, ma, mb, odds) - 1, N, ma, mb, odds)
    df_outer["a2"] = df_outer["p2"]+1 - nchypergeom_fisher.cdf(nchypergeom_fisher.ppf(1-df_outer["p2"], N, ma, mb, odds), N, ma, mb, odds)
    df_outer["alpha"] = np.minimum(df_outer["a1"],df_outer["a2"])
            
    
    get_df_outer(df_outer, df1_sup_sum, df2_sup_sum, df1_pNum, df2_pNum)
    return df_outer

def write_Excel_OR(df, outfile):
    writer = pd.ExcelWriter(diseasename+"/"+category+'/Comorbidity/' + outfile + '.xlsx')
    df.to_excel(writer, outfile, freeze_panes=(1, 1))  
    writer.save() 
    print('寫檔 : '+diseasename+"/"+category+'/Comorbidity/' + outfile + '.xlsx')

def zscore_to_tscore(z_score, sample_size, degrees_of_freedom):
    critical_value = stats.t.ppf((1 + 0.95) / 2, df=degrees_of_freedom)  # 95% confidence level
    t_score = z_score * (sample_size ** 0.5) / critical_value
    return t_score

def Intersection(readfile1, readfile2, df1_pNum, df2_pNum, ifOR, thresholdOR, PJI):
    df1 = pd.read_excel(readfile1 + '.xlsx')  # , dtype = {'pattern' : str})
    df2 = pd.read_excel(readfile2 + '.xlsx')  # , dtype = {'pattern' : str})

    df1_outer = pd.merge(df1, df2, on='pattern', how='outer')
    df2_outer = pd.merge(df2, df1, on='pattern', how='outer')

    # sup_sum
    df1_sup_sum = df1['#SUP'].sum()
    df2_sup_sum = df2['#SUP'].sum()

    df1_outer = get_df_outer_2(df1_outer, df1_sup_sum, df2_sup_sum, df1_pNum, df2_pNum)
    df2_outer = get_df_outer_2(df2_outer, df2_sup_sum, df1_sup_sum, df2_pNum, df1_pNum)

    output_outer = df2_outer#[['pattern','#SUP_x', 'No#SUP_x','#SUP_y', 'No#SUP_y',"OR_2"]]
    #output_outer.rename(columns={'pattern': 'ICD', '#SUP_x': 'Disease Suffer', 'No#SUP_x': 'Disease Non-Suffer', '#SUP_y': 'Health Suffer', 'No#SUP_y': 'Health Non-Suffer', 'OR_2':'Odds Ratio'}, inplace=True)
    #output_outer = output_outer.loc[(output_outer['Disease Suffer'] >= disease_count*sup_filter) & (output_outer['Odds Ratio'] >= ORthrehold)].sort_values(by='Odds Ratio', ascending=False)
    output_outer.to_excel(diseasename+"/"+category+'/Comorbidity/' + 'comorbidity feature set '+category+ '.xlsx', freeze_panes=(1, 1))  
    
    if ifOR != "":
        if PJI == "PJI":
            def threshold(x):
                if (x[ifOR+"_2"] < thresholdOR ) or x['#SUP_x'] <  disease_count * sup_filter:
                    return 1
                else:
                    return x['#SUP_x']
        elif PJI == "OPJI":
            def threshold(x):
                if (x[ifOR+"_2"] < thresholdOR ) or x['#SUP_x'] <  disease_count * sup_filter:
                #if x['#SUP_x'] <  disease_count * sup_filter or ( (x["alpha"] > 0.000001 ) and (x[ifOR+"_2"] < thresholdOR))
                    return 1
                else:
                    return x['#SUP_x']*x["OR_norm"]
            def norm_OR(df_outer, normalize):
                print(df_outer)
                normalize = df_outer.loc[(df_outer[ifOR+"_2"] >= thresholdOR ) & (df_outer["#SUP_x"] >= disease_count * sup_filter)]["OR_norm_tmp"].tolist()
                mean_norm = np.mean(normalize)
                std_norm = np.std(normalize)
                print(normalize)
                min_norm = min(normalize)
                z_norm = max(normalize)-min(normalize)
                tmp_list = []
                tmp_list_2 = []
                n = df_outer["OR_norm_tmp"].count()
                degrees_of_freedom = n - 1
                for (index,item) in enumerate(df_outer["OR_norm_tmp"].tolist()):
                    if df_outer[ifOR+"_2"][index] < thresholdOR or df_outer["#SUP_x"][index] < disease_count * sup_filter:
                        tmp_list.append(0)
                    else:
                        #tmp = round((item-mean_norm)/std_norm*10+10,2)
                        tmp = zscore_to_tscore((item-mean_norm)/std_norm, n, degrees_of_freedom)
                        tmp_list.append(tmp)
                        #tmp_list_2.append(tmp)
                '''tmp_list_2.sort()
                count = 0
                final = tmp_list_2[0]
                while final < 0:
                    final+=1
                    count+=1
                for (index,item) in enumerate(tmp_list):
                    if item != 0:
                        tmp_list[index] = item+count'''
                for (index,item) in enumerate(tmp_list):
                    if item < 0:
                        tmp_list[index] = 0.5
                return tmp_list
                    
        elif PJI == "APJI":
            def threshold(x):
                if x[ifOR+"_2"]< thresholdOR or  x['#SUP_x'] <  disease_count * sup_filter:
                    return 1
                else:
                    return x['#SUP_x']*x["alpha_norm"] 
            def norm_alpha(df_outer, normalize):
                tmp = (df_outer.loc[(df_outer[ifOR+"_2"] >= thresholdOR) &(df_outer["#SUP_x"] >= disease_count * sup_filter)]["alpha"]).tolist()
                tmp = [i for i in tmp if i != 0]
                min_threshold = min(tmp)
                print(min_threshold)
                print(min_threshold*1000)
                #normalize = (df_outer.loc[(df_outer["alpha"] <= 0.00000001 ) & (df_outer["#SUP_x"] >= disease_count * sup_filter)]["alpha_norm_tmp"]).tolist()
                normalize = (df_outer.loc[(df_outer[ifOR+"_2"] >= thresholdOR)  & (df_outer["alpha"] <= 0.3) &(df_outer["#SUP_x"] >= disease_count * sup_filter)]["alpha_norm_tmp"]).tolist()
                #normalize = [1-i for i in normalize] & (df_outer["alpha"] <= 0.000000001)
                df_outer.to_excel(diseasename+"/"+category+'/Comorbidity/' + 'comorbidity feature set_2222 '+category+ '.xlsx', freeze_panes=(1, 1))  
                min_norm = min(normalize)
                z_norm = max(normalize)-min(normalize)
                mean = np.mean(normalize)
                std_norm = np.std(normalize)
                tmp_list = []
                #tmp_list_2 = []
                for (index,item) in enumerate(df_outer["alpha_norm_tmp"].tolist()):
                    #if df_outer["alpha"][index] > 0.00000001 or df_outer["#SUP_x"][index] < disease_count * sup_filter:
                    if df_outer[ifOR+"_2"][index] < thresholdOR or df_outer["#SUP_x"][index] < disease_count * sup_filter:
                        tmp_list.append(0)
                    else:
                        tmp = round((item-mean)/std_norm*10+10,2)
                        tmp_list.append(tmp)
                        #tmp_list_2.append(tmp)
                for (index,item) in enumerate(tmp_list):
                    if item < 0:
                        tmp_list[index] = 0.5
                return tmp_list

        df1_outer_OR = df1_outer[(df1_outer['#SUP_x'] > 0)]
        df2_outer_OR = df2_outer[(df2_outer['#SUP_x'] > 0)]

        if PJI == "OPJI":
            normalize_or = df1_outer_OR[ifOR+"_2"].tolist()
            df1_outer_OR["OR_norm"] = pd.Series([(float(i)-np.mean(normalize_or))/np.std(normalize_or) for i in normalize_or])
            #df1_outer_OR["OR_norm"] =  norm_OR(df1_outer_OR, normalize_or)

            normalize_or = df2_outer_OR[ifOR+"_2"].tolist()
            df2_outer_OR["OR_norm_tmp"] = pd.Series([(float(i)-np.mean(normalize_or))/(np.std(normalize_or)) for i in normalize_or])
            df2_outer_OR["OR_norm"] =  norm_OR(df2_outer_OR, normalize_or)
            
        elif PJI == "APJI":
            tmp = 1-df1_outer_OR["alpha"]
            normalize_alpha = tmp.tolist()
            df1_outer_OR["alpha_norm_tmp"] =  pd.Series([(float(i)-np.mean(normalize_alpha))/(np.std(normalize_alpha)) for i in normalize_alpha])
            df1_outer_OR["alpha_norm"] = norm_alpha(df1_outer_OR, normalize_alpha)

            tmp = 1-df2_outer_OR["alpha"]
            normalize_alpha = tmp.tolist()
            df2_outer_OR["alpha_norm_tmp"] =  pd.Series([(float(i)-np.mean(normalize_alpha))/(np.std(normalize_alpha)) for i in normalize_alpha])#df2_outer_OR["alpha_norm"] = 1-df2_outer_OR["alpha"]##norm_alpha(df2_outer_OR, normalize_alpha)
            df2_outer_OR["alpha_norm"] = norm_alpha(df2_outer_OR, normalize_alpha)

            

        df1_outer_OR['#SUP_x'] = df1_outer_OR.apply(lambda row: threshold(row), axis=1)
        df2_outer_OR['#SUP_x'] = df2_outer_OR.apply(lambda row: threshold(row), axis=1)
        # sup_sum
        df1_sup_sum_OR = df1_outer_OR['#SUP_x'].sum()
        df2_sup_sum_OR = df2_outer_OR['#SUP_x'].sum()
        df2_outer_OR = get_df_outer_2(df2_outer_OR, df2_sup_sum_OR, df1_sup_sum_OR, df2_pNum, df1_pNum)
        df2_outer_OR.sort_values(by='#SUP_x', ascending=False)

        # (EXCEL)
        writer = pd.ExcelWriter(diseasename+"/"+category+'/Comorbidity/' + 'df21_outer_' + PJI + '_'+category+ '.xlsx')
        df2_outer_OR.to_excel(writer, 'df21_outerr_'+PJI+'_'+category, freeze_panes=(1, 1))  
        writer.save()  
        print('write excel : '+diseasename+"/"+category+'/Comorbidity/' + 'df21_outer_'+category+'_' + PJI + '.xlsx')
        return df2_outer_OR


def read_df_OR(filename):
    # read_excel # .loc # rename
    df_result = pd.read_excel(diseasename+"/"+category+'/Comorbidity/' + filename + '.xlsx')
    df_result = df_result.loc[:, ['#SUP_x', 'No#SUP_x', 'pattern']]
    df_result = df_result.rename(columns={ "#SUP_x": "#SUP", 'No#SUP_x':'No#SUP'})
    return df_result


def get_1_df_OR_Topdisease(df_result, outfile):

    df_result = df_result.loc[:, ['#SUP_x', 'No#SUP_x', 'pattern']]
    df_result = df_result.rename(columns={ "#SUP_x": "#SUP", 'No#SUP_x':'No#SUP'})
    # order # rank
    df_result['order'] = list(range(1, len(df_result) + 1))
    df_result['Rank'] = df_result['#SUP'].rank(ascending=False, method='dense')

    
    writer = pd.ExcelWriter(diseasename+"/"+category+'/Comorbidity/' + outfile + '.xlsx')
    df_result.to_excel(writer, outfile, freeze_panes=(1, 1))  
    writer.save() 
    print('write excel : '+diseasename+"/"+category+'/Comorbidity/' + outfile + '.xlsx')
    return df_result


if not os.path.isdir("./"+diseasename+"/"+category+"/Comorbidity"): os.mkdir("./"+diseasename+"/"+category+"/Comorbidity")
    
df_outer_OR = Intersection(diseasename+"/"+category+'/TopDisease/TopDisease_Normal_'+ category,diseasename+"/"+category+'/TopDisease/TopDisease_'+diseaseicd+'_'+ category,health_count,disease_count, "OR", ORthrehold, "PJI")
df_ORTopDisease_PJI = get_1_df_OR_Topdisease(df_outer_OR, category + '_ORTopDisease_PJI')
df_outer_OR = Intersection(diseasename+"/"+category+'/TopDisease/TopDisease_Normal_'+ category,diseasename+"/"+category+'/TopDisease/TopDisease_'+diseaseicd+'_'+ category,health_count,disease_count, "OR", ORthrehold, "OPJI")
df_ORTopDisease_OPJI = get_1_df_OR_Topdisease(df_outer_OR, category + '_ORTopDisease_OPJI')
df_outer_OR = Intersection(diseasename+"/"+category+'/TopDisease/TopDisease_Normal_'+ category,diseasename+"/"+category+'/TopDisease/TopDisease_'+diseaseicd+'_'+ category,health_count,disease_count, "OR", ORthrehold, "APJI")
df_ORTopDisease_APJI = get_1_df_OR_Topdisease(df_outer_OR, category + '_ORTopDisease_APJI')

def get_df_outer_3(df_outer, df1_sup_sum, df2_sup_sum):
    # PJ
    df_outer['XPJ'] = np.where((pd.isnull(df_outer['#SUP_x'])),
                               np.nan,
                               df_outer['#SUP_x'] / df1_sup_sum)
    df_outer['YPJ'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                               np.nan,
                               df_outer['#SUP_y'] / df2_sup_sum)
    df_outer['PJ'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                              df_outer['XPJ'],
                              (df_outer['XPJ'] + df_outer['YPJ']) / 2)
    df_outer['PJT'] = np.where((pd.isnull(df_outer['#SUP_y'])),
                               df_outer['#SUP_x'] / df1_sup_sum,
                               (df_outer['#SUP_x'] / df1_sup_sum + df_outer['#SUP_y'] / df2_sup_sum) / 2)
    return df_outer

#J、PJ
def pt1_Top_Disease(icd_list):

    df = pd.DataFrame(icd_list, columns={'pattern'}, index=icd_list)
    
    df['pattern'] = df['pattern'].astype(float) if '-' not in df['pattern'][0] else df['pattern']
    df['#SUP'] = 1
    # order # rank
    df['order'] = list_order = list(range(1, len(df) + 1))
    df['Rank'] = df['#SUP'].rank(ascending=False, method='dense')
    df = df.reindex(['order', 'Rank', '#SUP', 'No#SUP', 'pattern'], axis=1)
    return df

def Intersection_TWO(df1_1, df1_2, df1_3, df2):
    #PJ
    df1 = df1_1
    df1_outer = pd.merge(df1, df2, on='pattern', how='outer')
    df2_outer = pd.merge(df2, df1, on='pattern', how='outer')
    df_inner = pd.merge(df1, df2, on='pattern')  

    # sup_sum
    df1_sup_sum = df1['#SUP'].sum() 
    df2_sup_sum = df2['#SUP'].sum()
    df1_outer = get_df_outer_3(df1_outer, df1_sup_sum, df2_sup_sum)
    df2_outer = get_df_outer_3(df2_outer, df2_sup_sum, df1_sup_sum)
    df_inner = get_df_outer_3(df_inner, df1_sup_sum, df2_sup_sum)

    # Jaccarrd index、Weighted Jaccard index
    J_intersection = len(df_inner)
    J_union = len(df1) + len(df2) - J_intersection
    J_index = J_intersection / J_union 

    W_intersection = df_inner['PJ'].sum()
    W_union = df1_outer['PJ'].sum() + df2_outer['PJ'].sum() - W_intersection
    W_index = W_intersection / W_union 

    #OPJI
    df1 = df1_2
    df1_outer = pd.merge(df1, df2, on='pattern', how='outer')
    df2_outer = pd.merge(df2, df1, on='pattern', how='outer')
    df_inner = pd.merge(df1, df2, on='pattern')  

    # sup_sum
    df1_sup_sum = df1['#SUP'].sum() 
    df2_sup_sum = df2['#SUP'].sum()
    df1_outer = get_df_outer_3(df1_outer, df1_sup_sum, df2_sup_sum)
    df2_outer = get_df_outer_3(df2_outer, df2_sup_sum, df1_sup_sum)
    df_inner = get_df_outer_3(df_inner, df1_sup_sum, df2_sup_sum)
    # Odds Ratio Weighted Jaccard index
    ORW_intersection = df_inner['PJ'].sum()
    ORW_union = df1_outer['PJ'].sum() + df2_outer['PJ'].sum() - ORW_intersection
    ORW_index = ORW_intersection / ORW_union 

    #APJI
    df1 = df1_3
    df1_outer = pd.merge(df1, df2, on='pattern', how='outer')
    df2_outer = pd.merge(df2, df1, on='pattern', how='outer')
    df_inner = pd.merge(df1, df2, on='pattern')  

    # sup_sum
    df1_sup_sum = df1['#SUP'].sum() 
    df2_sup_sum = df2['#SUP'].sum()
    df1_outer = get_df_outer_3(df1_outer, df1_sup_sum, df2_sup_sum)
    df2_outer = get_df_outer_3(df2_outer, df2_sup_sum, df1_sup_sum)
    df_inner = get_df_outer_3(df_inner, df1_sup_sum, df2_sup_sum)
    # Odds Ratio Weighted Jaccard index
    AlphaW_intersection = df_inner['PJ'].sum()
    AlphaW_union = df1_outer['PJ'].sum() + df2_outer['PJ'].sum() - AlphaW_intersection
    AlphaW_index = AlphaW_intersection / AlphaW_union 

    return J_index, W_index, ORW_index, AlphaW_index

def OR_Intersection_pt1(PJ_file, OPJ_file, APJ_file, outfile, list_pticd):
    list_J = []
    list_PJ = []
    list_OPJ = []
    list_APJ = []
    
    df1_1 = PJ_file
    df1_2 = OPJ_file
    df1_3 = APJ_file
    for pticd in list_pticd:
        if pticd:  #pticd
            df2 = pt1_Top_Disease(pticd)
            J_index, W_index, ORW_index, AlphaW_index = Intersection_TWO(df1_1, df1_2, df1_3, df2)

            list_J.append(J_index)
            list_PJ.append(W_index)
            list_OPJ.append(ORW_index)
            list_APJ.append(AlphaW_index)
        else:
            list_J.append(0)
            list_PJ.append(0)
            list_OPJ.append(0)
            list_APJ.append(0)
    
    np_J_sum = np.array(list_J)
    print('J', 'mean', np.mean(np_J_sum), 'std', np.std(np_J_sum))
    np_PJ_sum = np.array(list_PJ)
    print('PJ', 'mean', np.mean(np_PJ_sum), 'std', np.std(np_PJ_sum))
    np_OPJ_sum = np.array(list_OPJ)
    print('OPJ', 'mean', np.mean(np_OPJ_sum), 'std', np.std(np_OPJ_sum))

    # list_J
    outfile_J = diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_' + outfile + '_J'
    np.save(outfile_J, list_J)
    np.savetxt(outfile_J+'.csv', list_J, delimiter=',')
    # list_PJ
    outfile_PJ = diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_' + outfile + '_PJ'
    np.save(outfile_PJ, list_PJ)
    np.savetxt(outfile_PJ+'.csv', list_PJ, delimiter=',')
    # list_OPJ
    outfile_OPJ = diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_' + outfile + '_OPJ'
    np.save(outfile_OPJ, list_OPJ)
    np.savetxt(outfile_OPJ+'.csv', list_OPJ, delimiter=',')
    # list_APJ
    outfile_APJ = diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_' + outfile + '_APJ'
    np.save(outfile_APJ, list_APJ)
    np.savetxt(outfile_APJ+'.csv', list_APJ, delimiter=',')


if not os.path.isdir("./"+diseasename+"/"+category+"/Comorbidity"): os.mkdir("./"+diseasename+"/"+category+"/Comorbidity")

list_ptid_icd_health = list(np.load(diseasename+"/"+category+'/GetIcdList/health_control_list_'+category+'_ptid_icd.npy',allow_pickle=True))
list_ptid_icd_disease = list(np.load(diseasename+"/"+category+'/GetIcdList/disease_list_'+category+'_ptid_icd.npy',allow_pickle=True)) 

OR_Intersection_pt1(df_ORTopDisease_PJI, df_ORTopDisease_OPJI, df_ORTopDisease_APJI, 'list_'+ category +'_ptid_icd_disease', list_ptid_icd_disease)
OR_Intersection_pt1(df_ORTopDisease_PJI, df_ORTopDisease_OPJI, df_ORTopDisease_APJI, 'list_'+ category +'_ptid_icd_health', list_ptid_icd_health) 