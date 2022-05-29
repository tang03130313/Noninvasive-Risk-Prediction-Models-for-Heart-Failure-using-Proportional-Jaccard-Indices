import pandas as pd
import datetime
import sys  

# min_0_Als、min_1_Als、min_Als
def min_disease_Def(ptid, disease, min_disease):
    # min_0_Als、min_1_Als
    min_0_disease = min(disease['0']) if disease['0'] else 0
    min_1_disease = min(disease['1']) if disease['1'] else 0

    # 
    if min_0_disease != 0 and min_1_disease != 0:
        if min_0_disease > min_1_disease:
            min_disease = min_1_disease
        else:
            min_disease = min_0_disease
    elif min_0_disease == 0:
        min_disease = min_1_disease
    elif min_1_disease == 0:
        min_disease = min_0_disease
    min_disease = datetime.datetime.strptime(str(min_disease), '%Y%m%d').date()  

    return  min_disease

def get_df_sup_2(dict_sup_ptid, total):
    dict_sup = {}
    for icd in dict_sup_ptid:
        dict_temp = {}
        tmp_len = len(dict_sup_ptid[icd])
        dict_temp['#SUP'] = tmp_len  
        dict_temp['No#SUP'] = total - tmp_len  
        dict_temp['pattern'] = icd
        dict_sup[icd] = dict_temp
    df_sup = pd.DataFrame(dict_sup) 
    df_sup = df_sup.T
    df_sup = df_sup.sort_values(by=['#SUP'], ascending=[0])

    # order # rank
    df_sup['order'] = list_order = list(range(1, len(df_sup) + 1))
    df_sup['Rank'] = df_sup['#SUP'].rank(ascending=False, method='dense')
    df_sup = df_sup.reindex(['order', 'Rank', '#SUP', 'No#SUP', 'pattern'], axis=1)

    return df_sup
list_mid = []
def minToMid(test_min):
    global list_mid
    if not list_mid:
        df = pd.read_csv("./ICD9/ICD9_mid.csv", encoding= 'utf8')
        list_mid = list(filter(lambda index: index[0] != 'V', df['ICD'].to_list()))

    for mid_icd in list_mid:
        if test_min >= int(mid_icd[0:3]) and test_min <= int(mid_icd[3:6]):
            return mid_icd
    print(test_min, False)
    return -1

def get_dict_sup_ptid(ptid, icd, dict_sup_ptid, category):
    if category == "individual" or category == "group":
        # individual dict_min_sup_ptid
        if dict_sup_ptid.__contains__(icd):
            dict_sup_ptid[icd].append(ptid)
        else:
            dict_sup_ptid[icd] = [ptid]
    return dict_sup_ptid


# {ptid : [icd1, icd2, ...]}
def get_dict_ptid_icd_2(ptid, icd, dict_ptid_icd, category):
    if category == "individual":
        # individual
        if ptid in dict_ptid_icd:
            dict_ptid_icd[ptid]['ICD'].append(icd)
        else:
            dict_ptid_icd[ptid] = {}
            dict_ptid_icd[ptid]['ICD'] = [icd]
    else:
        # group
        mid_icd = minToMid(int(icd))
        if mid_icd != -1:
            if ptid in dict_ptid_icd:
                if mid_icd not in dict_ptid_icd[ptid]['ICD']: 
                    dict_ptid_icd[ptid]['ICD'].append(mid_icd)
            else:
                dict_ptid_icd[ptid] = {}
                dict_ptid_icd[ptid]['ICD'] = [mid_icd]
    return dict_ptid_icd


