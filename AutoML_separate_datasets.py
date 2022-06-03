import numpy as np
import pandas as pd
import os 
import shutil 
import json
import datetime
import math
import plotly.express as px
import yaml
import sys
import xlrd
from shutil import copyfile

import TopDiseaseFunc
from config import category,health_count,disease_count,diseasename, diseaseicd, configname, ORthrehold, kfoldtimes, yaml_data

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def run() :
    #---------------------get_icd------------------------
    #os.system('python GetIcdList.py')   use autoget_separate_datasets.py generate subsets datasets, do not need to run again
    #print("-------------GetIcdList.py finish-------------")
    os.system('python TopDisease.py')
    print("-------------TopDisease.py finish-------------")
    #--------------------Comorbidity----------------------
    print('--------------------Comorbidity----------------------')
    os.system('python Comorbidity.py') 
    print("-------------Comorbidity finish-------------")
    #----------------------ComorbidityEveryOneKfold----------------------
    os.system('python ComorbidityML.py')
    print("-------------ComorbidityML finish-------------")
    copyfile("./config/"+configname+'.yaml', diseasename+'/'+ category + "/" + configname +'.yaml')
    copyfile("./Comorbidity.py", "./"+diseasename+"/"+category+"/Comorbidity/Comorbidity.py")

def save(output_name_dir,output_name):
    src = "./"+diseasename+"/"+category
    dest = output_name_dir+output_name
    copy_and_overwrite(src,dest)
    print("copy folders done")

    src = output_name_dir+output_name+"/ComorbidityML/PPT_"+category+"_standard.xlsx" 
    dest = output_name_dir+"PPT "+output_name+ ".xlsx" 
    copyfile(src, dest)
    print("copy final file done")

def copy_all(datasets_name):
    global years
    src = "./"+diseasename+"/datasets/"+category+'/'+years+'/'+datasets_name+"/"+configname+".yaml"
    dest = "./config/"+configname+'.yaml'
    copyfile(src, dest)
    dest =  "./"+diseasename+"/"+category+"/"+configname+'.yaml'
    copyfile(src, dest)
            
    print("copy config done")
    src = "./"+diseasename+"/datasets/"+category+'/'+years+'/'+datasets_name+"/GetIcdList"
    dest = "./"+diseasename+"/"+category+"/GetIcdList"
    copy_and_overwrite(src, dest)
    print("copy geticdlist done")

def system(datasets_name, output_state):
    copy_all(datasets_name)
    # change_support()
    output_name_dir = "./"+diseasename+"/"+"_results/"+output_file_name+"/"+category+"/"+years+"/"
    output_name = category+" "+years+" "+output_state+" OR"+str(ORthrehold)+" "+str(kfoldtimes)+"-fold"
    run()
    save(output_name_dir,output_name)

def change_year(year):
    with open("./config/"+configname+'.yaml', "r") as stream: yaml_data = yaml.safe_load(stream) 
    yaml_data["inputs"]["date_interval_flag"] = year  
    with open("./config/"+configname+'.yaml', 'w') as file: documents = yaml.dump(yaml_data, file)

def change_support():
    with open("./config/"+configname+'.yaml', "r") as stream: yaml_data = yaml.safe_load(stream) 
    yaml_data["inputs"]["sup_filter"] = 0.05  
    with open("./config/"+configname+'.yaml', 'w') as file: documents = yaml.dump(yaml_data, file)

output_file_name = diseasename+"_results_version1_paper"
if not os.path.isdir("./"+diseasename): os.mkdir("./"+diseasename)
if not os.path.isdir("./"+diseasename+"/"+category): os.mkdir("./"+diseasename+"/"+category)
if not os.path.isdir("./"+diseasename+"/"+"_results/"): os.mkdir("./"+diseasename+"/"+"_results/")
if not os.path.isdir("./"+diseasename+"/"+"_results/"+output_file_name): os.mkdir("./"+diseasename+"/"+"_results/"+output_file_name)
if not os.path.isdir("./"+diseasename+"/"+"_results/"+output_file_name+"/"+category): os.mkdir("./"+diseasename+"/"+"_results/"+output_file_name+"/"+category)

for_loop = 3
years = ""

for i in range(1,for_loop+1):
    years = str(i)+"y"
    print(years)
    change_year(i)
    #ori
    datasets_name = "_ori"
    output_state = "ori "
    system(datasets_name, output_state)
    # sex F
    state_sex = "F"
    datasets_name = "sex "+state_sex+" "+years
    output_state = "sex"+state_sex
    system(datasets_name, output_state)
    # sex M 
    state_sex = "M"
    datasets_name = "sex "+state_sex+" "+years
    output_state = "sex"+state_sex
    system(datasets_name, output_state)
    # age up
    state_age = 0
    output_age = "up" if state_age == 0 else "down"
    age = 65
    datasets_name = "age "+str(age)+output_age+" "+years
    output_state = "age"+str(age)+output_age
    system(datasets_name, output_state)
    # age down
    state_age = 1
    output_age = "up" if state_age == 0 else "down"
    age = 65
    datasets_name = "age "+str(age)+output_age+" "+years
    output_state = "age"+str(age)+output_age
    system(datasets_name, output_state)