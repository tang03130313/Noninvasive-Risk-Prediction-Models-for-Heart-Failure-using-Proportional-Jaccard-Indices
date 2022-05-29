
import yaml
import sys
with open("./config/config_188.yaml", "r") as stream: yaml_data = yaml.safe_load(stream)  # 335
configname = yaml_data["inputs"]["configname"]
category = yaml_data["inputs"]["category"]
diseasename = yaml_data["inputs"]["diseasename"]
diseaseicd = yaml_data["inputs"]["diseaseicd"]
diseasejson = yaml_data["inputs"]["diseasejson"]
noramaljson = yaml_data["inputs"]["noramaljson"]
ORthrehold = yaml_data["inputs"]["ORthrehold"]
sup_filter = yaml_data["inputs"]["sup_filter"]
date_interval_flag = yaml_data["inputs"]["date_interval_flag"]
health_endyear = yaml_data["inputs"]["health_endyear"]
hospitialized = yaml_data["inputs"]["hospitialized"]
if category != "individual" and category != "group":
    sys.exit("category error")
    
disease_count = yaml_data["variables"]["disease_count"]
health_count = yaml_data["variables"]["health_count"]
health_control_count = yaml_data["variables"]["health_control_count"]
kfoldtimes = yaml_data["variables"]["kfoldtimes"]
