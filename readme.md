# Using-Proportional-Jaccard-Indices-for-Identifying-Comorbidity-Patterns
# How to run
1. create conda environment in Python 3.8.13 (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
conda create --name test python=3.8
conda activate test
```
2. install requirements packages
```
pip install -r requirements.txt
```
3. add origin json file to ./OriginalData (or use GenerateRadomJson.py to generate random file )
```
python GenerateRadomJson.py
```
4. revise the parameter in config/config_configname.yaml
5. config.py set which yaml file want to use
6. start system
```
python index.py
```
7. diseasename folder will show the results

# Python files
## Flow Chart
![image](https://github.com/tang03130313/Using-Proportional-Jaccard-Indices-to-Identify-Comorbidity-Patterns-of-Heart-Failure/blob/main/images/flowchart.png)
### GenerateRadomJson.py
+ generate random  json file for an example
### GetIcdList  
+ Sorting out data from json file (The json files are the example with ICD-9-CM)
+ Transfering to csv file
### TopDisease
+ Counting the co-occurence frequency of the experimental group and the control group
### Comorbidity
+ The classification codes of all comorbidity diseases were analyzed by evaluating corresponding odds ratios (ORs) associated with target disease
+ Processing data by minimum odds ratio threshold and add pseudo-counts
+ Construct a comorbidity feature set
+ Calculating three similarity measurements including Jaccard index(JI), Proportional Jaccard index(PJI), Odds Ratio Proportional Jaccard index(OPJI), Alpha Proportional Jaccard index (APJI) (the comorbidity feature set vs the experiment group and the comorbidity feature set vs the control group)
### ComorbidityML
+ Constructing four machine learning 
+ logistic regression (LR)
+ support vector classifier (SVC) 
+ random forest (RF) 
+ eXtreme gradient boosting (XGBoost)
### AutoGet
+ auto generate different subsets
### AutoML
+ use the datasets from AutoGet.py run the whole system and save the results to the specific folder
### AutoBoxPlots
+ auto summarize the specific folder of different models and draw the different box plots 

# config detail
### inputs
+ ORthrehold: odds ratio threshold
+ category: individual stand for individual level, group stand for group level(take ICD-9-CM for an example)
+ configname: this file name
+ date_interval_flag: disease data trace back interval
+ diseaseicd: this target disease icd
+ diseasejson: disease json file name
+ diseasename: disease names
+ health_endyear: health control group endyear (ex. end year = 2010 -> A birth 195503 -> 2010-1935 = 55 years old)
+ hospitialized: 0 stand for icd outpatient, 1 stand for icd emergency and hospitalization
+ noramaljson: health json file name
+ supthreshold: subject number filter (by experimental group)
### variables
+ disease_count: disease group count
+ health_control_count: health group count(for ml datasets)
+ health_count: health group count
+ kfoldtimes: k of kfold p.s. will use K ≈ log (n) and n/K > 3d (n : the sample size, d : the number of parameters, and a natural logarithm of base e)

# Json file detail
+ This is a sample of json file from random python file (not real data)
```
"00cb13asfsd1680fd9ac281a5f505055361": {
        "BirthDay": 197503,
        "BirthYear": 1975,
        "DeathDay": NaN,
        "DeathYear": NaN,
        "sex": "M"
        "ICD": {
            "349": {
                "0": [
                    20100420
                ],
                "1": [
                    20100520,
                    20100620,
                ]
            },
            ...
        }
```
#### target disease code
+ In target disease json, every cases will a least one record of target disease code
+ In no target disease json, every cases will not contain any target disease code
#### age
+ In target disease json, we use first diagnosis of target disease for patient's age (ex. A 'icd':[20010810, 20020701], birth 195506 -> 2001 - 1955 = 46)
+ In no target disease json, we use (end year - birth year) for patient's age (ex. end year = 2010 -> B birth 195503 -> 2010-1935 = 55 years old)

# License
Copyright © 2022, [Yueh Tang](https://github.com/tang03130313).
Released under the [MIT license](LICENSE).
