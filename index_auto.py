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

#---------------------AutoGet_separate_datasets------------------------
os.system('python AutoGet_separate_datasets.py')   
#print("-------------AutoML_separate_datasets.py finish-------------")

os.system('python AutoML_separate_datasets.py')
print("-------------AutoML_separate_datasets.py finish-------------")

#--------------------AutoBoxPlots----------------------
print('--------------------AutoBoxPlots----------------------')
#os.system('python AutoBoxPlots.py') 
print("-------------AutoBoxPlots finish-------------")

