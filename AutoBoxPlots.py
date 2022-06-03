import glob
import os 
import numpy as np
import pandas as pd
import plotly.express as px
import yaml
import sys

from config import category,diseasename,configname

osname = diseasename+"_results_version1_paper"
dirPathPattern = "./"+diseasename+"/_results/"+osname+"/*/*/*.xlsx"
excel_files = glob.glob(dirPathPattern)
print(len(excel_files))

JI_list = ["JI", "PJI", "OPJI", "APJI"]
ML_list = ['LR','SVC','RF','XGB']
parameters_list = ['AUC','Accuracy','F1Score','Recall','Precision']
column_names = ["para", "score", "category", "years", "datasets", "JI", "ML"]
df_outputs = pd.DataFrame(columns=column_names)
for file in excel_files:
    df = pd.read_excel(file, sheet_name="result").T
    df.columns = df.iloc[0] 
    df = df.iloc[1: , :]
    filename_arr = file.split("\\")[-1].split()
    for para in parameters_list:
        index = 0
        for ML in ML_list:
            for JI in JI_list:
                new_row = {"para": para ,"score": df[para][index], "category": filename_arr[1], "years":  filename_arr[2], "datasets":  filename_arr[3], "JI": JI, "ML": ML}
                df_outputs = df_outputs.append(new_row, ignore_index=True)
                index += 1
if not os.path.isdir("./"+diseasename+"/_results/"+osname+"/boxplots/"): os.mkdir("./"+diseasename+"/_results/"+osname+"/boxplots/")
df_outputs.to_excel("./"+diseasename+"/_results/"+osname+"/boxplots/summarized boxplot.xlsx")  

outputs_list = ["ML", "category", "years", "datasets"]
for para in parameters_list:
    if not os.path.isdir("./"+diseasename+"/_results/"+osname+"/boxplots/"+para): os.mkdir("./"+diseasename+"/_results/"+osname+"/boxplots/"+para)
    tmp_df = df_outputs.loc[(df_outputs["para"] == para)]
    tmp_df = tmp_df.rename(columns={'score': para})
    for outputs_para in outputs_list:
        fig = px.box(tmp_df, x=outputs_para, y=para, color="JI",title=diseasename+"/"+outputs_para+" "+para)
        fig.update_traces(quartilemethod="linear") 
        fig.write_image("./"+diseasename+"/_results/"+osname+"/boxplots/"+para+"/"+diseasename+" "+outputs_para+" "+para+".png")
        fig.write_html("./"+diseasename+"/_results/"+osname+"/boxplots/"+para+"/"+diseasename+" "+outputs_para+" "+para+".html")
