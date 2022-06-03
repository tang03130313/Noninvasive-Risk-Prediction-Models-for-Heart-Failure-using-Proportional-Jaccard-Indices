import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import random
import math

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy import optimize

from sklearn.linear_model import Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
#from sklearn.externals import joblib
import joblib
import datetime

import yaml
from config import category,disease_count,diseasename,kfoldtimes, health_control_count, configname, yaml_data
import os
if not os.path.isdir("./"+diseasename+"/"+category+"/ComorbidityML"): os.mkdir("./"+diseasename+"/"+category+"/ComorbidityML")
if not os.path.isdir("./"+diseasename+"/"+category+"/ComorbidityML/model"): os.mkdir("./"+diseasename+"/"+category+"/ComorbidityML/model")

totalNum = disease_count

def kfold_ppt(filename, X_list, y_list, X_name_list):
    writer_PPT = pd.ExcelWriter(
        diseasename+"/"+category+'/ComorbidityML\\PPT_' + filename + ".xlsx")
    df_name = pd.DataFrame(X_name_list, columns={'X_name_list'}, index=range(1, len(X_name_list) + 1))
    df_name.to_excel(writer_PPT, 'file', freeze_panes=(1, 1)) 
    
    df_empty = pd.DataFrame({})
    df_empty.to_excel(writer_PPT, 'result', freeze_panes=(1, 1)) 

    # models
    clf_LG = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    clf_SVC = SVC(gamma='auto', probability=True)
    clf_RF = RandomForestClassifier(max_depth=2, random_state=0)
    clf_XG = XGBClassifier(max_depth=5, learning_rate=0.1, num_class=2, n_estimators=160, silent=True, objective='multi:softmax')
    clf_list = [clf_LG, clf_SVC, clf_RF, clf_XG]
    
    # write excel results
    df_clf = pd.DataFrame(clf_list, columns={'clf_list'}, index=range(1, len(clf_list) + 1))
    df_clf.to_excel(writer_PPT, 'file', freeze_panes=(1, 1), startcol=5) 
    
    parameters_list = ['AUC','Accuracy','F1Score','Recall','Precision']
    list_boxplot_all = {}#[]
    for para in parameters_list: list_boxplot_all[para]=[]

    df_result = pd.DataFrame({})
    for index_clf in range(len(clf_list)):
        for index_X in range(len(X_list)):
            clf = clf_list[index_clf]
            X = X_list[index_X]
            y = y_list[index_X]
            
            df = kfold_training(clf, X, y, kfoldtimes, totalNum, index_clf, index_X)
            for para in parameters_list: list_boxplot_all[para].append(df.loc[ para , : ][str(kfoldtimes) + ' fold'])
            df.to_excel(writer_PPT, str(index_clf) + '+' + str(index_X), freeze_panes=(1, 1))  
            print(str(index_clf) + '+' + str(index_X), '====================================')
            print('clf', clf)
            print('x name', X_name_list[index_X])
            
            df_temp = pd.DataFrame({})
            df_temp = df.loc[:, ['mean']]
            df_temp = df_temp.rename(columns={"mean": str(index_clf) + '+' + str(index_X)})
            if not len(df_result):
                df_result = df_temp
            else:
                df_result = pd.merge(df_result, df_temp, left_index=True, right_index=True)
            df_temp = pd.DataFrame({})
            #df_temp = df_temp.rename(columns={"list_riskRR": str(index_clf) + '+' + str(index_X)})
    box_plot(list_boxplot_all)
    df_result.to_excel(writer_PPT, 'result', freeze_panes=(1, 1))  

            
    writer_PPT.save() 
    print("寫檔 : "+diseasename+"/"+category+"/ComorbidityML\\PPT.xlsx")

# models boxplot 
def box_plot(list_boxplot_all):
    print(X_name_list)
    models = ['LR','SVC','RF','XGB']
    for para in list_boxplot_all:
        result = []
        for i in range(0,len(models)):
            for j in range(0,len(X_name_list)):
                count = 0
                rows = eval(list_boxplot_all[para][int(len(result)/kfoldtimes)])
                for row in rows:
                    result.append([(count)%kfoldtimes,X_name_list[j],models[i],row])
                    count += 1
        df = pd.DataFrame(result, columns = ['id','type','model',para])
        fig = px.box(df, x="model", y=para, color="type",title=diseasename+"/"+category+" "+para)
        fig.update_traces(quartilemethod="linear") 
        fig.write_html(diseasename+"/"+category+"/ComorbidityML/boxplot/boxplot_"+para+".html")

    
def kfold_training(clf, X, y, kfoldtimes, totalNum, index_clf, index_X):
    # initial variable
    k = kfoldtimes
    total = totalNum
    
    
    first = 0
    last = 0
    mod = total % k

    listThreshold = []
    listModelScore = []
    listPrecision = []  # correctDiseaseCount / PredictDiseaseNum 
    listRecall = []  # correctDiseaseCount / DiseaseDataNum 
    listSpecificity = [] # uncorrectHeaelthCount / HealthDataNum  
    listAccuracy = []  # (correctDiseaseCount + correctHeaelthCount) / len(X_test)  
    listF1Score = []  # 2 / ((1 / Precision) + (1 / Recall))
    listF1Score_all = [] 
    listAUC = []  # AUC
    listLRPositive = [] 
    listLRNegative = [] 

    # kfold
    for i in range(1, k + 1):
        first = last
        if i > k - mod: 
            last = first + (total // k) + 1
        else: 
            last = first + (total // k)

        X_train = np.delete(X, list(range(first, last)) + list(range(total + first, total + last)), 0)  # Disease + health
        y_train = np.delete(y, list(range(first, last)) + list(range(total + first, total + last)), 0)  # Disease + health
        X_test = np.array(list(X[first:last]) + list(X[total + first:total + last]))  # Disease + health
        y_test = np.array(list(y[first:last]) + list(y[total + first:total + last]))  # Disease + health

        # Training
        clf = clf.fit(X_train, y_train)
        
        # Validation
        Precision, Recall, Specificity, Accuracy, F1Score, LRPositive, LRNegative = validation(clf, X_test, y_test)
        
        # plot roc get auc
        lr_auc = roc_auc(clf, X_test, y_test)

        # list to results
        listModelScore.append(round(clf.score(X_train, y_train), 3))
        listPrecision.append(round(Precision, 3))  # correctDiseaseCount / PredictDiseaseNum  
        listRecall.append(round(Recall, 3))  # correctDiseaseCount / DiseaseDataNum 
        listSpecificity.append(round(Specificity, 3)) # uncorrectHeaelthCount / HealthDataNum 
        listAccuracy.append(round(Accuracy, 3))  # (correctDiseaseCount + correctHeaelthCount) / len(X_test)  
        listF1Score.append(round(F1Score, 3))
        listAUC.append(round(lr_auc, 3))
        listLRPositive.append(round(LRPositive, 3))
        listLRNegative.append(round(LRNegative, 3))

    models = ['LR','SVC','RF','XGB']
    joblib.dump(clf, "./"+diseasename+"/"+category+"/ComorbidityML/model/"+X_name_list[index_X]+"_"+models[index_clf]+".model")
    
    # K Fold Result
    print("---------------- K Fold Result ----------------")
    print("Threshold\t", round(sum(listThreshold) / k, 4), listThreshold)
    print("ModelScore\t", round(sum(listModelScore) / k, 4), listModelScore)
    print("Precision\t", round(sum(listPrecision) / k, 4), listPrecision)
    print("Recall\t\t", round(sum(listRecall) / k, 4), listRecall)
    print("Specificity\t", round(sum(listSpecificity) / k, 4), listSpecificity)
    print("Accuracy\t", round(sum(listAccuracy) / k, 4), listAccuracy)
    print("F1Score\t\t", round(sum(listF1Score) / k, 4), listF1Score)
    print("AUC\t\t", round(sum(listAUC) / k, 4), listAUC)
    print("listLRPositive\t\t", round(sum(listLRPositive) / k, 4), listLRPositive)
    print("listLRNegative\t\t", round(sum(listLRNegative) / k, 4), listLRNegative)

    print('training model', clf)
    dict_kfoldResult = {'Threshold':[round(sum(listThreshold) / k, 3), listThreshold], 
                'ModelScore':[round(sum(listModelScore) / k, 3), listModelScore],
                'Precision':[round(sum(listPrecision) / k, 3), listPrecision],
                'Recall':[round(sum(listRecall) / k, 3), listRecall],
                'Specificity':[round(sum(listSpecificity) / k, 3), listSpecificity],
                'Accuracy':[round(sum(listAccuracy) / k, 3), listAccuracy],
                'F1Score':[round(sum(listF1Score) / k, 3), listF1Score],
                'AUC':[round(sum(listAUC) / k, 3), listAUC], 
                'LR+':[round(sum(listLRPositive) / k, 3), listLRPositive], 
                'LR-':[round(sum(listLRNegative) / k, 3), listLRNegative]
                
                       }
    df_kfoldResult = pd.DataFrame(data=dict_kfoldResult, index=['mean', str(kfoldtimes)+' fold']).astype(str)
    df_all = pd.DataFrame(data = df_kfoldResult.iloc[0] + df_kfoldResult.iloc[1])
    df_mean = pd.DataFrame(data = df_kfoldResult.iloc[0])
    df_fold = pd.DataFrame(data = df_kfoldResult.iloc[1])
    df = pd.merge(df_all, df_mean, left_index=True, right_index=True)
    df = pd.merge(df, df_fold, left_index=True, right_index=True)
    return df

# return Precision, Recall, Specificity, Accuracy, F1Score, LRPositive, LRNegative
def validation(clf, X_test, y_test):
    # Accuracy check # Disease 1 health0
    correctDiseaseCount = 0
    uncorrectDiseaseCount = 0
    correctHeaelthCount = 0
    uncorrectHeaelthCount = 0
    for i in range(0, len(X_test)):
        if clf.predict(X_test[i].reshape(-1,1)):  # Disease
            if y_test[i] == 1:
                correctDiseaseCount += 1
            else:
                uncorrectDiseaseCount += 1
        else:  # health
            if y_test[i] == 0:
                correctHeaelthCount += 1
            else:
                uncorrectHeaelthCount += 1

    TotalDataNum = len(X_test)
    DiseaseDataNum = np.sum(y_test == 1)
    HealthDataNum = np.sum(y_test == 0)
    PredictDiseaseNum = np.sum(clf.predict(X_test) == 1) # np.sum(X_test >= threshold) # np.sum(clf.predict(X_test) == 1)
    PredictHealthNum = np.sum(clf.predict(X_test) == 0) # np.sum(X_test < threshold)

    Precision = correctDiseaseCount / PredictDiseaseNum  
    Recall = correctDiseaseCount / DiseaseDataNum  
    Specificity = correctHeaelthCount / HealthDataNum 
    Accuracy = (correctDiseaseCount + correctHeaelthCount) / len(X_test)  
    F1Score = 2 / ((1 / Precision) + (1 / Recall))
    LRPositive = Recall / (1-Specificity) 
    LRNegative = (1-Recall) / Specificity 

    return Precision, Recall, Specificity, Accuracy, F1Score, LRPositive, LRNegative

# plot roc get auc
# return lr_auc
def roc_auc(clf, X_test, y_test):
    # roc curve and auc
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    ns_probs = [0 for _ in range(len(y_test))]
    lr_probs = clf.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    # calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, ns_threshold = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, lr_threshold = roc_curve(y_test, lr_probs)
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic AUC : ' + str(round(lr_auc, 3)))
    # axis labels
    plt.xlabel('FDiseasee Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # save the plot
    plt.savefig("./"+diseasename+"/"+category+"/ComorbidityML/Plot Logistic.png")

    return lr_auc

# ifStandard >> true > standard, fDiseasee > no standard
def preprocess_X(X, ifStandard):
    if ifStandard:
        X = preprocessing.scale(X)
    X = np.array([[i] for i in X])
    return X

def remove_all_zero(list):
    return [i for i in list if i != 0]

tmp =  round(math.log(disease_count + health_control_count))
if tmp != kfoldtimes:
    kfoldtimes = tmp
    yaml_data["variables"]["kfoldtimes"] = tmp
    with open("./config/"+configname+'.yaml', 'w') as file: documents = yaml.safe_dump(yaml_data, file)
# Data Generator

OR_Intersection_pt1_list_ptid_icd_Disease_J = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_disease_J.npy'))
OR_Intersection_pt1_list_ptid_icd_Disease_PJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_disease_PJ.npy'))
OR_Intersection_pt1_list_ptid_icd_Disease_OPJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_disease_OPJ.npy'))
OR_Intersection_pt1_list_ptid_icd_Disease_APJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_disease_APJ.npy'))
OR_Intersection_pt1_list_ptid_icd_health_J = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_health_J.npy'))
OR_Intersection_pt1_list_ptid_icd_health_PJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_health_PJ.npy'))
OR_Intersection_pt1_list_ptid_icd_health_OPJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_health_OPJ.npy'))
OR_Intersection_pt1_list_ptid_icd_health_APJ = list(np.load(diseasename+"/"+category+'/Comorbidity/OR_Intersection_pt1_list_'+ category +'_ptid_icd_health_APJ.npy'))


random.seed(0)
X_J = random.sample(OR_Intersection_pt1_list_ptid_icd_Disease_J, disease_count) + random.sample(OR_Intersection_pt1_list_ptid_icd_health_J, health_control_count)
random.seed(0)
X_PJ = random.sample(OR_Intersection_pt1_list_ptid_icd_Disease_PJ, disease_count) + random.sample(OR_Intersection_pt1_list_ptid_icd_health_PJ, health_control_count)
y = np.hstack((np.ones(disease_count), np.zeros(disease_count)))
random.seed(0)
X_OPJ = random.sample(OR_Intersection_pt1_list_ptid_icd_Disease_OPJ, disease_count) + random.sample(OR_Intersection_pt1_list_ptid_icd_health_OPJ, health_control_count)
random.seed(0)
X_APJ = random.sample(OR_Intersection_pt1_list_ptid_icd_Disease_APJ, disease_count) + random.sample(OR_Intersection_pt1_list_ptid_icd_health_APJ, health_control_count)

from pandas import DataFrame
tmp = []
tmp.append(X_J)
tmp.append(X_PJ)
tmp.append(X_OPJ)
tmp.append(X_APJ)
tmp.append(y)
df = DataFrame (tmp).transpose()
df.columns = ['X_J','X_PJ','X_OPJ', 'X_APJ','Y']
if not os.path.isdir("./"+diseasename+"/"+category+"/ComorbidityML/datasets"): os.mkdir("./"+diseasename+"/"+category+"/ComorbidityML/datasets")
df.to_csv("./"+diseasename+"/"+category+"/ComorbidityML/datasets/datasets.csv")

if not os.path.isdir("./"+diseasename+"/"+category+"/ComorbidityML/boxplot"): os.mkdir("./"+diseasename+"/"+category+"/ComorbidityML/boxplot")

X_name_list = ['X_J', 'X_PJ', 'X_OPJ', 'X_APJ']
X_list = [preprocess_X(X_J, True), preprocess_X(X_PJ, True), preprocess_X(X_OPJ, True), preprocess_X(X_APJ, True)]
y_list = [y, y, y, y]
kfold_ppt(category + '_standard', X_list, y_list, X_name_list)