import numpy as np
import pandas as pd
import os
import json
import datetime
from datetime import date
import math
import plotly.express as px
import yaml
import sys
import TopDiseaseFunc

from config import category,yaml_data,diseasename,diseaseicd,diseasejson,configname, noramaljson, hospitialized, date_interval_flag, health_endyear
if not os.path.isdir("./"+diseasename): os.mkdir("./"+diseasename)
if not os.path.isdir("./"+diseasename+"/"+category): os.mkdir("./"+diseasename+"/"+category)
if not os.path.isdir("./"+diseasename+"/"+category+"/GetIcdList"): os.mkdir("./"+diseasename+"/"+category+"/GetIcdList")
if not os.path.isdir("./"+diseasename+"/"+category+"/GetIcdList/chart"): os.mkdir("./"+diseasename+"/"+category+"/GetIcdList/chart")

def DataExtractOp(op, filename = diseasejson):
    print('DataExtractOp', '---------------------')
    readfile = os.path.abspath('.') + '/OriginalData/' + filename

    with open(readfile + '.json', 'r') as f: 
        data = json.load(f)
        df = pd.DataFrame(data)
    print("讀檔 : " + readfile + ".json")

    ptids_op = []  
    ptids_no_target_disease = [] 
    for ptid in data: 
        if data[ptid]['ICD'].__contains__(diseaseicd):  
            if data[ptid]['ICD'][diseaseicd][op]:         
                ptids_op.append(ptid) 

        else: 
            ptids_no_target_disease.append(ptid)
    print('len(ptids_op)', len(ptids_op), 'len(ptids_no_target_diseas)', len(ptids_no_target_disease))
    
    print('DataExtractOp', 'End---------------------')
    return ptids_op

if hospitialized == 0:
    specific_ptids = DataExtractOp('0')  # op, filename  op 0(outpatient)、1(emergancy and hospitalization)
elif hospitialized == 1:
    specific_ptids = DataExtractOp('1')  

readfile_disease = os.path.abspath('.') + '/OriginalData/'+diseasejson
readfile_health = os.path.abspath('.') + '/OriginalData/' + noramaljson

currentDateTime = datetime.datetime.now()
nowdate = currentDateTime.date()
now_year = nowdate.strftime("%Y")
now_year = int(now_year)

# 讀取數據
with open(readfile_disease + '.json', 'r') as f:
    data_disease = json.load(f)
print("讀檔 : " + readfile_disease + ".json")

with open(readfile_health + '.json', 'r') as f:
    data_health = json.load(f)
print("讀檔 : " + readfile_health + ".json")
data_health_index = []
for pid in data_health:
    data_health_index.append(pid)
print('.json\t', 'disease人數:', len(data_disease), 'health人數:', len(data_health), 'date_interval_flag:', date_interval_flag)
 
def calculateAge(birthDate, today):
    age = today.year - birthDate.year -((today.month, today.day) <(birthDate.month, birthDate.day))
    return age

dict_sup_ptid_disease = {} # {ptid : [icd1, icd2, ...]}
dict_ptid_icd_disease = {} # {ptid : [icd1, icd2, ...]}
dict_sup_ptid_health = {}   # {icd : [ptid1, ptid2, ....]} # sup
dict_ptid_icd_health = {}   # {ptid : [icd1, icd2, ...]}
dict_sup_ptid_health_control = {}
dict_ptid_icd_health_control = {} # ML
health_index = 0
disease_age_list = []
disease_sex_list = []


# disease group
disease_age_list = []
disease_sex_list = []
disease_id_list = []
disease_type_list = []
count = 0
for ptid in data_disease:  
    count += 1
    disease = []
    min_disease = 0
    print(count)
    if data_disease[ptid]['ICD'].__contains__(diseaseicd) and ptid in specific_ptids:  
        print(ptid)
        disease = data_disease[ptid]['ICD'][diseaseicd]
    
        min_disease = TopDiseaseFunc.min_disease_Def(ptid, disease, min_disease)
        
        before_ymin_disease = min_disease + datetime.timedelta(-365)*date_interval_flag

        before_y_min_disease = int(before_ymin_disease.strftime('%Y%m%d'))
        if min_disease != 0: min_disease = int(min_disease.strftime('%Y%m%d'))
        date_interval = [before_y_min_disease, min_disease]
        for icd in data_disease[ptid]['ICD']:
            if icd != diseaseicd:
                date_list = data_disease[ptid]['ICD'][icd]['0'] + data_disease[ptid]['ICD'][icd]['1']
                new_date_list = [i for i in date_list if
                                 i >= date_interval[0] and i <= date_interval[1]]  
                if len(new_date_list):  
                    dict_ptid_icd_disease = TopDiseaseFunc.get_dict_ptid_icd_2(ptid, icd,dict_ptid_icd_disease, category)
        if dict_ptid_icd_disease.__contains__(ptid):
            min_disease = str(min_disease)
            birth = str(data_disease[ptid]['BirthDay'])
            birth_date = date(int(birth[0:4]), int(birth[4:6]), 1)
            disease_date = date(int(min_disease[0:4]), int(min_disease[4:6]), 2)
            tmp_age = calculateAge(birth_date,disease_date)
            print(tmp_age)
            disease_age_list.append(tmp_age)
            dict_ptid_icd_disease[ptid]['age'] = tmp_age
            dict_ptid_icd_disease[ptid]['sex'] = data_disease[ptid]['sex']
            disease_sex_list.append(data_disease[ptid]['sex'])
            disease_id_list.append(ptid)
            disease_type_list.append(0)
df_disease_feature = pd.DataFrame(list(zip(disease_id_list,disease_age_list,disease_sex_list,disease_type_list)), columns = ["id","age","sex","type"])
df_disease_feature_m = df_disease_feature[df_disease_feature["sex"]=="M"]
df_m = df_disease_feature_m.groupby('age')['id'].nunique()
df_m.to_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_disease_M.csv")
df_disease_feature_m = df_disease_feature[df_disease_feature["sex"]=="F"]
df = df_disease_feature_m.groupby('age')['id'].nunique()
df.to_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_disease_F.csv")

fig = px.box(df_disease_feature, x="type", y="age",color="sex",title=diseasename+" disease_age")
fig.update_traces(quartilemethod="linear") 
fig.write_html("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_disease.html")


# Control group
health_age_list = []
health_sex_list = []
health_id_list = []
health_type_list = []
count = 0
error_sex = []
error_age = []
count_di = 0
count_add = 0

for ptid in data_health:  
    count += 1
    disease = []
    min_disease = 0
    tmp_age = health_endyear - data_health[ptid]['BirthYear']-1
    health_age_list.append(tmp_age)
    health_sex_list.append(data_health[ptid]['sex'])
    health_id_list.append(ptid)
    health_type_list.append(0)

df_health_feature = pd.DataFrame(list(zip(health_id_list,health_age_list,health_sex_list,health_type_list)), columns = ["ptid","age","sex","type"])

df_health_feature_m = df_health_feature[df_health_feature["sex"]=="M"]
df = df_health_feature_m.groupby('age')['ptid'].nunique()
df.to_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_health_M.csv")
df_health_feature_m = df_health_feature[df_health_feature["sex"]=="F"]
df = df_health_feature_m.groupby('age')['ptid'].nunique()
df.to_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_health_F.csv")

fig = px.box(df_health_feature, x="type", y="age",color="sex",title=diseasename+" health_age")
fig.update_traces(quartilemethod="linear")
fig.write_html("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_health.html")

exception_1_count = 0
exception_2_count = 0
count_yes_1 = 0
dict_ptid_icd_health = {}
# first get control group for ml validation
for index,row in df_disease_feature.iterrows():
    bool_check = False
    df_data_health_sex = df_health_feature[df_health_feature["sex"]==row['sex']]
    age = row['age']
    age_ori = row['age']
    sex = row['sex']
    if len(df_data_health_sex) == 0:
        ptid = df_health_feature.iloc[0]["ptid"]
        dict_ptid_icd_health[ptid] = {}
        dict_ptid_icd_health[ptid]['ICD']= []
        dict_ptid_icd_health[ptid]["age"] = row['age']
        dict_ptid_icd_health[ptid]["sex"] = row['sex']
        df_health_feature = df_health_feature.drop(df_health_feature.iloc[[0]].index)
        exception_2_count += 1
        bool_check = True
    else:
        state = 0
        while(not bool_check):
            df_data_health_select = df_data_health_sex[(df_data_health_sex["age"]==age)]
            lenaaa = len(df_data_health_select)
            for index_2, row_2 in df_data_health_select.iterrows():
                ptid = row_2["ptid"]
                month = str(data_health[ptid]['BirthDay']%100).zfill(2)
                end_year = health_endyear
                date_1 = int(str(end_year-date_interval_flag)+month+"01")
                date_2 = int(str(end_year)+month+"31")
                date_interval = [date_1, date_2]
                print(date_interval)
                for icd in data_health[ptid]['ICD']:
                    if icd != diseaseicd:
                        date_list = data_health[ptid]['ICD'][icd]['0'] + data_health[ptid]['ICD'][icd]['1']
                        new_date_list = [i for i in date_list if
                                        i >= date_interval[0] and i <= date_interval[1]] 
                        if len(new_date_list): 
                            dict_ptid_icd_health = TopDiseaseFunc.get_dict_ptid_icd_2(ptid, icd, dict_ptid_icd_health, category)
                               
                if ptid in dict_ptid_icd_health:
                    dict_ptid_icd_health[ptid]['age'] = age
                    dict_ptid_icd_health[ptid]['sex'] = row['sex']
                    df_health_feature = df_health_feature.drop(index_2)
                    count_yes_1 += 1
                    bool_check = True
                    break
            if bool_check:
                break
            elif state == 0:
                if (abs(age_ori-age) <= 2):
                    age = age-1 
                else:
                    age = age_ori
                    state = 1
            elif state == 1:
                if (abs(age_ori-age) <= 2):
                    age = age+1 
                else:
                    age = age_ori
                    state = 2
            else:
                state = 0
                ptid = df_data_health_sex.iloc[0]["ptid"]
                dict_ptid_icd_health[ptid] = {}
                dict_ptid_icd_health[ptid]['ICD']= []
                dict_ptid_icd_health[ptid]["age"] = row['age']
                dict_ptid_icd_health[ptid]["sex"] = row['sex']
                df_health_feature = df_health_feature.drop(df_data_health_sex.iloc[[0]].index)
                exception_1_count += 1
                bool_check = True
                break
dict_ptid_icd_health_control = dict_ptid_icd_health.copy()

exception_2_count = 0
# get others control group
for index,row in df_health_feature.iterrows():
    age = row['age']
    sex = row['sex']
    ptid = row['ptid']
    print(age,sex)
    month = str(data_health[ptid]['BirthDay']%100).zfill(2)
    end_year = health_endyear
    date_1 = int(str(end_year-date_interval_flag)+month+"01")
    date_2 = int(str(end_year)+month+"31")
    date_interval = [date_1, date_2]
    print(date_interval)
    for icd in data_health[ptid]['ICD']:
        if icd != diseaseicd:
            date_list = data_health[ptid]['ICD'][icd]['0'] + data_health[ptid]['ICD'][icd]['1']
            new_date_list = [i for i in date_list if
                            i >= date_interval[0] and i <= date_interval[1]]

            if len(new_date_list): 
                dict_ptid_icd_health = TopDiseaseFunc.get_dict_ptid_icd_2(ptid, icd, dict_ptid_icd_health, category)
                
    if ptid in dict_ptid_icd_health:
        df_health_feature = df_health_feature.drop(index)
        dict_ptid_icd_health[ptid]["age"] = row['age']
        dict_ptid_icd_health[ptid]["sex"] = row['sex']
        count_yes_1 += 1
    else:
        exception_2_count += 1

for ptid in dict_ptid_icd_disease:
    for icd in dict_ptid_icd_disease[ptid]['ICD']:
        dict_sup_ptid_disease = TopDiseaseFunc.get_dict_sup_ptid(ptid, icd, dict_sup_ptid_disease, category)
for ptid in dict_ptid_icd_health:
    for icd in dict_ptid_icd_health[ptid]['ICD']:
        dict_sup_ptid_health = TopDiseaseFunc.get_dict_sup_ptid(ptid, icd, dict_sup_ptid_health, category)   
for ptid in dict_ptid_icd_health_control:
    for icd in dict_ptid_icd_health_control[ptid]['ICD']:
        dict_sup_ptid_health_control = TopDiseaseFunc.get_dict_sup_ptid(ptid, icd, dict_sup_ptid_health_control, category)

# output
def save(dict_ptid_icd,dict_sup_ptid,type):
    list_ptid_icd = []
    for ptid in dict_ptid_icd:
        list_ptid_icd.append(dict_ptid_icd[ptid]['ICD'])
    yaml_data["variables"][type+"_count"] = len(list_ptid_icd)
    with open("./config/"+configname+'.yaml', 'w') as file: documents = yaml.safe_dump(yaml_data, file)
    file.close()

    if not os.path.isdir("./"+diseasename+"/"+category+"/GetIcdList"): os.mkdir("./"+diseasename+"/"+category+"/GetIcdList")
    np.save(diseasename+"/"+category+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd', list_ptid_icd)
    df_list = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(list_ptid_icd) })
    df_list = df_list.T
    df_list.to_csv(diseasename+"/"+category+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd.csv')
    with open(diseasename+"/"+category+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd'+'.json', 'w') as f:
        json.dump(dict_ptid_icd, f)

    print("寫檔 : "+diseasename+"/"+category+"/GetIcdList/"+type+"_list_"+ category +"_ptid_icd")
    with open(diseasename+"/"+category+'/GetIcdList/'+type+'_dict_'+ category +'_sup_ptid.json', "w") as f_data:
        json.dump(dict_sup_ptid, f_data, indent=4)
    print("寫檔 : "+diseasename+"/"+category+"/GetIcdList/"+type+"dict_"+ category +"_sup_ptid.json")

save(dict_ptid_icd_disease,dict_sup_ptid_disease,"disease")
save(dict_ptid_icd_health_control,dict_sup_ptid_health_control,"health_control")
save(dict_ptid_icd_health,dict_sup_ptid_health,"health")


df1 = pd.read_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_disease_F.csv")
df2 = pd.read_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_health_F.csv")
df_ = pd.merge(df1,df2, how='inner')
df_['sum'] = df_['id']+df_['ptid']
df_f1 = df_[['age','sum']]
list = []
sex = []
for i in range(0,len(df_f1)):
    list.append('0')
    sex.append('F')
df_f1['type'] = list
df_f1['sex'] = sex
df1 = pd.read_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_disease_M.csv")
df2 = pd.read_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_health_M.csv")
df_ = pd.merge(df1,df2, how='inner')
df_['sum'] = df_['id']+df_['ptid']
df_f2 = df_[['age','sum']]
list = []
sex = []
for i in range(0,len(df_f2)):
    list.append('0')
    sex.append('M')
df_f2['type'] = list
df_f2['sex'] = sex
df_f = pd.merge(df_f1,df_f2, how='outer')
df_f.to_csv("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_all_sex.csv")
fig = px.box(df_f, x="type", y="age",color="sex",title=diseasename+" disease_age")
fig.update_traces(quartilemethod="linear")
fig.write_html("./"+diseasename+"/"+category+"/GetIcdList/chart/"+diseasename+"_all_sex.html")
