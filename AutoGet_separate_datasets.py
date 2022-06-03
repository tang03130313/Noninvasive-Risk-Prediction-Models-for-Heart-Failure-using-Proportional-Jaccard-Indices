import numpy as np
import pandas as pd
import os
import json
import datetime
import math
import plotly.express as px
import yaml
import sys
import shutil 
from shutil import copyfile

import TopDiseaseFunc
from config import category, yaml_data,health_count,disease_count,diseasename, diseaseicd,configname, diseasejson, noramaljson

def read(years, type):
    with open(diseasename+"/datasets/"+category+'/'+years+'/_ori/GetIcdList/'+type+'_list_'+ category +'_ptid_icd'+'.json') as f:
        dict_ptid_icd = json.load(f)
    return dict_ptid_icd

def separate_sex(dict_ptid_icd, years, type, sex):
    dict_ptid_icd = {ptid: {x: dict_ptid_icd.get(ptid).get(x) for x in dict_ptid_icd.get(ptid) } for ptid in dict_ptid_icd if dict_ptid_icd.get(ptid).get("sex")==sex}
    return  dict_ptid_icd

def separate_age(dict_ptid_icd, years, type, state, age):
    if state == 0:
        dict_ptid_icd = {ptid: {x: dict_ptid_icd.get(ptid).get(x) for x in dict_ptid_icd.get(ptid) } for ptid in dict_ptid_icd if dict_ptid_icd.get(ptid).get("age") >= age}
    elif state == 1:
        dict_ptid_icd = {ptid: {x: dict_ptid_icd.get(ptid).get(x) for x in dict_ptid_icd.get(ptid) } for ptid in dict_ptid_icd if dict_ptid_icd.get(ptid).get("age") < age}
    return  dict_ptid_icd

def save(dict_ptid_icd,type, name):
    dict_sup_ptid = {}
    for ptid in dict_ptid_icd:
        for icd in dict_ptid_icd[ptid]['ICD']:
            dict_sup_ptid = TopDiseaseFunc.get_dict_sup_ptid(ptid, icd, dict_sup_ptid, category)
    list_ptid_icd = []
    for ptid in dict_ptid_icd:
        list_ptid_icd.append(dict_ptid_icd[ptid]['ICD'])
        
    if not os.path.isdir(diseasename+"/datasets/"+category+'/'+years+'/'+name): os.mkdir(diseasename+"/datasets/"+category+'/'+years+'/'+name)
    if not os.path.isdir(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/'): os.mkdir(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/')
    
    yaml_data["variables"][type+"_count"] = len(list_ptid_icd)
    with open(diseasename+"/datasets/"+category+'/'+years+'/'+name+"/"+configname+'.yaml', 'w') as file: documents = yaml.dump(yaml_data, file)
    file.close()

    np.save(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd', list_ptid_icd)
    df_list = pd.DataFrame({ i:pd.Series(value) for i, value in enumerate(list_ptid_icd) })
    df_list = df_list.T
    df_list.to_csv(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd.csv')
    with open(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/'+type+'_list_'+ category +'_ptid_icd'+'.json', 'w') as f:
        json.dump(dict_ptid_icd, f)

    print("寫檔 : "+diseasename+"/datasets/"+category+'/'+years+'/'+name+"/GetIcdList/"+type+"_list_"+ category +"_ptid_icd")
    with open(diseasename+"/datasets/"+category+'/'+years+'/'+name+'/GetIcdList/'+type+'_dict_'+ category +'_sup_ptid.json', "w") as f_data:
        json.dump(dict_sup_ptid, f_data, indent=4)
    print("寫檔 : "+diseasename+"/datasets/"+category+'/'+years+'/'+name+"/GetIcdList/"+type+"dict_"+ category +"_sup_ptid.json")

def copy_config():
    copyfile("./config/"+configname+'.yaml', diseasename+'/'+ category + "/" + configname +'.yaml')

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def sex_system(state_sex, dict_ptid_icd_disease, dict_ptid_icd_health, dict_ptid_icd_health_control):
    global years
    output_name = "sex "+state_sex+" "+years
    dict_ptid_icd_disease = separate_sex(dict_ptid_icd_disease, years, "disease", state_sex)
    dict_ptid_icd_health = separate_sex(dict_ptid_icd_health, years, "health", state_sex)
    dict_ptid_icd_health_control = separate_sex(dict_ptid_icd_health_control, years, "health_control", state_sex)
    save(dict_ptid_icd_disease,"disease",output_name)
    save(dict_ptid_icd_health_control,"health_control",output_name)
    save(dict_ptid_icd_health,"health",output_name)

def age_system(state_age, age, dict_ptid_icd_disease, dict_ptid_icd_health, dict_ptid_icd_health_control):
    global years
    output_age = output_age = "up" if state_age == 0 else "down"
    output_name = "age "+str(age)+output_age+" "+years
    dict_ptid_icd_disease = separate_age(dict_ptid_icd_disease, years, "disease", state_age,age)
    dict_ptid_icd_health = separate_age(dict_ptid_icd_health, years, "health", state_age,age)
    dict_ptid_icd_health_control = separate_age(dict_ptid_icd_health_control, years, "health_control", state_age,age)
    save(dict_ptid_icd_disease,"disease",output_name)
    save(dict_ptid_icd_health_control,"health_control",output_name)
    save(dict_ptid_icd_health,"health",output_name)

def change_year(year):
    with open("./config/"+configname+'.yaml', "r") as stream: yaml_data = yaml.safe_load(stream) 
    yaml_data["inputs"]["date_interval_flag"] = year  
    with open("./config/"+configname+'.yaml', 'w') as file: documents = yaml.dump(yaml_data, file)


if not os.path.isdir("./"+diseasename): os.mkdir("./"+diseasename)
if not os.path.isdir("./"+diseasename+"/"+category): os.mkdir("./"+diseasename+"/"+category)
if not os.path.isdir(diseasename+"/datasets"): os.mkdir(diseasename+"/datasets")
if not os.path.isdir(diseasename+"/datasets/"+category): os.mkdir(diseasename+"/datasets/"+category)

for_loop = 3 # lead-time interval years
years = ""
for i in range(1,for_loop+1):
    years = str(i)+"y"
    print(years)
    if not os.path.isdir(diseasename+"/datasets/"+category+'/'+years): os.mkdir(diseasename+"/datasets/"+category+'/'+years)
    change_year(i)
    os.system('python GetIcdList.py') 
    copy_config()
    src = "./"+diseasename+"/"+category+"/GetIcdList"
    dest = "./"+diseasename+"/datasets/"+category+'/'+years+'/'+"_ori"+"/GetIcdList"
    copy_and_overwrite(src, dest)
    copyfile("./config/"+configname+'.yaml', "./"+diseasename+"/datasets/"+category+'/'+years+'/'+"_ori/" + configname +'.yaml')
    dict_ptid_icd_disease = read(years, "disease")
    dict_ptid_icd_health = read(years, "health")
    dict_ptid_icd_health_control = read(years, "health_control")
    dict_ptid_icd_disease_ori = dict_ptid_icd_disease.copy()
    dict_ptid_icd_health_ori = dict_ptid_icd_health.copy()
    dict_ptid_icd_health_control_ori = dict_ptid_icd_health_control.copy()
    state_sex = "F"
    sex_system(state_sex, dict_ptid_icd_disease_ori, dict_ptid_icd_health_ori, dict_ptid_icd_health_control_ori)
    state_sex = "M"
    sex_system(state_sex, dict_ptid_icd_disease_ori, dict_ptid_icd_health_ori, dict_ptid_icd_health_control_ori)
    age = 65
    state_age = 0
    age_system(state_age, age, dict_ptid_icd_disease_ori, dict_ptid_icd_health_ori, dict_ptid_icd_health_control_ori)
    state_age = 1
    age_system(state_age, age, dict_ptid_icd_disease_ori, dict_ptid_icd_health_ori, dict_ptid_icd_health_control_ori)

