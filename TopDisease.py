import pandas as pd
import json
import datetime
import TopDiseaseFunc
import os
import yaml
from config import category,health_count,disease_count,diseasename,diseaseicd,sup_filter

def Top_Disease(outfile):

    with open(diseasename+"/"+category+'/GetIcdList/disease_dict_'+ category +'_sup_ptid.json', 'r') as f:
        dict_sup_ptid = json.load(f) 

    df_sup = TopDiseaseFunc.get_df_sup_2(dict_sup_ptid, disease_count)  

    df_sup = df_sup[df_sup['#SUP'] > 0]
    print(df_sup)

    # (EXCEL)
    writer = pd.ExcelWriter(
        os.path.abspath('') + '/'+diseasename+"/"+category+'/TopDisease/' + 'TopDisease_' + outfile + '_'+ category +'.xlsx')
    df_sup.to_excel(writer, category, freeze_panes=(1, 1)) 
    writer.save() 
    print("寫檔 : " + os.path.abspath('') + "/"+diseasename+"/"+category+"/TopDisease/"+  str(len(df_sup)) + 'TopDisease_' + outfile + "_"+ category +".xlsx")


def Top_Disease_Health(outfile):

    with open(diseasename+"/"+category+'/GetIcdList/health_dict_'+ category +'_sup_ptid.json', 'r') as f:
        dict_sup_ptid = json.load(f)
    
    df_sup = TopDiseaseFunc.get_df_sup_2(dict_sup_ptid, health_count)  
    
    df_sup = df_sup[df_sup['#SUP'] > 0]

    # (EXCEL)
    writer = pd.ExcelWriter(
        os.path.abspath('') + '/'+diseasename+"/"+category+'/TopDisease/' + 'TopDisease_' + outfile + '_'+ category +'.xlsx')
    df_sup.to_excel(writer, category, freeze_panes=(1, 1))  
    writer.save()  
    print("Write : " + os.path.abspath('') + "/"+diseasename+"/"+category+"/TopDisease/"+  str(len(df_sup)) + 'TopDisease_' + outfile + "_"+ category +".xlsx")

if not os.path.isdir("./"+diseasename+"/"+category+"/TopDisease"): os.mkdir("./"+diseasename+"/"+category+"/TopDisease")

Top_Disease(diseaseicd)  
Top_Disease_Health('Normal')