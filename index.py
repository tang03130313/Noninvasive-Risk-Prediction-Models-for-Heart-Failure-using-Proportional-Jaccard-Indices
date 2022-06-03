import pandas as pd
import numpy as np
import json
import datetime
import TopDiseaseFunc
import os
import xlrd
import yaml
from shutil import copyfile

from config import category,health_count,disease_count,diseasename,configname

if not os.path.isdir("./"+diseasename): os.mkdir("./"+diseasename)
if not os.path.isdir("./"+diseasename+"/"+category): os.mkdir("./"+diseasename+"/"+category)

#---------------------get_icd------------------------
os.system('python GetIcdList.py')   
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
