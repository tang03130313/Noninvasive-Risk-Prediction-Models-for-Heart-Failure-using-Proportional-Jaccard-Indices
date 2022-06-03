#from random import Random
import random
import math
import json
#random = Random()

disease_json_name = "sample_222"   #disease_json_name
health_json_name = "sample_no222"  #health_json_name
id_random_num = 32                 #id_random_num
disease_count = 1000               #experimental group count
disease_code_num = 60            #each patient have 0-30 disease (include 222)
health_count = 2000                #experimental group count
health_code_num = 15               #each patient have 0-15 disease (do not have 222)
target_id = "222"                  #target disease code
target_id_diagnosis = 1            #target disease diagnosis (0 for outpatient, 1 for emergancy and hospitalization)
codes_total_num = 999              #disease code standard (1-999)
endyear = 2010                     #health control group age standard (endyear - birthyear = age)
date_interval = 3                  #lead-time interval , 3 for three year

sex_list = 'MF'

def random_id(num):
    str = ''
    chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
    length = len(chars) - 1
    for i in range(num):
        str+=chars[random.randint(0,length)]
    return str

def random_target_date_array(min_date_year, min_date):
    global target_id, target_id_diagnosis, disease_code_num
    arr1 = []
    arr2 = []
    if target_id_diagnosis == 0:
        arr1.append(min_date)
    elif target_id_diagnosis == 1:
        arr2.append(min_date)
    for i in range(random.randint(0,10)):
        date = random.randint(min_date_year,2018)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        while date <= min_date: date = random.randint(min_date_year,2018)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        arr1.append(date)
    for i in range(random.randint(0,10)):
        date = random.randint(min_date_year,2018)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        while date <= min_date: date = random.randint(min_date_year,2018)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        arr2.append(date)
    arr1.sort()
    arr2.sort()
    dict = {"0": arr1, "1": arr2}
    return dict
def random_date_array(min_date_year, min_date):
    global target_id, disease_code_num
    arr1 = []
    arr2 = []
    for i in range(random.randint(0,5)):
        date = random.randint(min_date_year-date_interval,min_date_year)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        while date >= min_date: date = random.randint(min_date_year-3,min_date_year)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        arr1.append(date)
    for i in range(random.randint(0,5)):
        date = random.randint(min_date_year-date_interval,min_date_year)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        while date >= min_date: date = random.randint(min_date_year-3,min_date_year)*10000 + random.randint(1,12)*100 + random.randint(1,28)
        arr2.append(date)
    arr1.sort()
    arr2.sort()
    dict = {"0": arr1, "1": arr2}
    return dict
def disease_patient(age, sex):
    global id_random_num, target_id, disease_code_num, codes_total_num, disease_dict
    id = random_id(id_random_num)
    while id in disease_dict: id = random_id(id_random_num)
    min_date_year = random.randint(1998,2013)
    min_date_month = random.randint(1,12)
    min_date = min_date_year*10000 + min_date_month*100 + random.randint(1,28)
    birth_year = min_date_year-age
    birth_day = birth_year*100 + random.randint(1,12)
    print(min_date)
    dict = {id: {"BirthDay": birth_day, "BirthYear": birth_year, "DeathDay": math.nan, "DeathYear": math.nan,"sex": sex,"ICD":{}}}
    dict[id]["ICD"][target_id] = random_target_date_array(min_date_year, min_date)
    for i in range(random.randint(0,disease_code_num)):
        tmp_id = random.randint(1,codes_total_num)
        while tmp_id in dict: tmp_id = random.randint(1,codes_total_num)
        dict[id]["ICD"][tmp_id] = random_date_array(min_date_year, min_date)
    return dict    
def health_patient(age, sex):
    global id_random_num, target_id, health_code_num, codes_total_num, health_dict
    id = random_id(id_random_num)
    while id in health_dict: id = random_id(id_random_num)
    birth_year = endyear-age
    birth_month = random.randint(1,12)
    birth_day = birth_year*100 + birth_month
    min_date = endyear*10000 + birth_month*100 + random.randint(1,28)
    print(min_date)
    dict = {id: {"BirthDay": birth_day, "BirthYear": birth_year, "DeathDay": math.nan, "DeathYear": math.nan,"sex": sex,"ICD":{}}}
    for i in range(random.randint(0,health_code_num)):
        tmp_id = random.randint(1,codes_total_num)
        while (tmp_id in dict or tmp_id == int(target_id)) : tmp_id = random.randint(1,codes_total_num)
        dict[id]["ICD"][tmp_id] = random_date_array(endyear, min_date)
    return dict 
disease_dict = {}
health_dict = {}

differ = health_count//disease_count
differ_2 = health_count/disease_count
disease_age_list = []
disease_sex_list = []
for i in range(disease_count):
    age = random.randint(20,90)
    disease_age_list.append(age)
    sex = sex_list[random.randint(0,1)]
    disease_sex_list.append(sex)
    disease_dict.update(disease_patient(age, sex))
    for j in range(differ):
        health_dict.update(health_patient(age, sex))
if differ != differ_2:
    index = 0
    for j in range(health_count-(disease_count*differ)):
        age = disease_age_list[index]
        sex = disease_sex_list[index]
        health_dict.update(health_patient(age, sex))     
        index += 1
print(len(disease_dict))
print(len(health_dict))

with open("./OriginalData/"+disease_json_name+".json", "w") as outfile:
    json.dump(disease_dict, outfile)

with open("./OriginalData/"+health_json_name+".json", "w") as outfile:
    json.dump(health_dict, outfile) 