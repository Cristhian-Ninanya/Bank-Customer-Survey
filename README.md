# Bank Customers Survey - Marketing for Term Deposit

**Data Set Information:**

https://archive.ics.uci.edu/ml/datasets/bank+marketing#

The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

**Target**

The goal is to implement 3 classifiers that allow us to evaluate and compare their metrics and chose the best one.

## 1.Attribute Information:

**Input variables:**

* 1.age (numeric)
* 2.job: type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services") 
* 3.marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
* 4.education (categorical: "unknown","secondary","primary","tertiary")
* 5.default: has credit in default? (binary: "yes","no")
* 6.balance: average yearly balance, in euros (numeric) 
* 7.housing: has housing loan? (binary: "yes","no")
* 8.loan: has personal loan? (binary: "yes","no")
* 9.contact: contact communication type (categorical: "unknown","telephone","cellular") 
* 10.day: last contact day of the month (numeric)
* 11.month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
* 12.duration: last contact duration, in seconds (numeric)
* 13.campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* 14.pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
* 15.previous: number of contacts performed before this campaign and for this client (numeric)
* 16.poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")

**Output variable (desired target)**
* 17.y:has the client subscribed a term deposit? (binary: "yes","no")

## 2.Set virtual environment

use terminal CMD (VSCODE)
```cmd
C:\ML_scikitlearn>python get-pip.py
C:\ML_scikitlearn>python -m virtualenv entorno
C:\ML_scikitlearn>entorno\Scripts\activate.bat
(entorno) C:\ML_scikitlearn>pip install -r requirements.txt
C:\ML_scikitlearn>cd entorno
C:\ML_scikitlearn\entorno>cd Scripts
C:\ML_scikitlearn\entorno\Scripts>activate.bat
(entorno) C:\ML_scikitlearn\entorno\Scripts>pip list
```
## 3.File Organization (VS Code)

```
├── __pycache__        <- Created automatically
├── .vscode            <- Created automatically
├── dataset            <- Store all DataSet
│   ├── data_bank_customer_survey.csv
│
├── Lib
│   ├── site-packages  <- packages for virtual enviroment (pip install -r requirements.txt)
│
├── results
│   ├── figure1.jpg    <- Unbalanced DataSet (before)
│   ├── figure2.jpg    <- Balanced DataSet (after)
│   ├── figure3.jpg    <- Compare unbalanced dataset metrics
│   ├── figure4.jpg    <- Compare balanced dataset metrics
│   ├── log.txt        <- Manually saved log after execute main.py (optional)
│
├── .gitignore         <- ignore during git commit (optional)
├── get-pip.py         <- 
├── main.py            <- Main Project
├── models.py          <- Scripts for classification models, trainning , metrics.
├── pyvenv.cfg         <- Configuration parameters for virtual enviroment
├── requirements.txt   <- Libraries for virtual enviroment
├── utils.py           <- Aditional scripts for main.py

```

## 3.Set path to load DataSet

Relative Path: dataset/data_bank_customer_survey.csv
General Path: C:\ML_scikitlearn\entorno\dataset\data_bank_customer_survey.csv
```python
#================= 1.Data Preprocessing ===========================
#----------------- 1.1 Load Data  -------------------------------
    #Load and print Data set:
    df = utl.load_from_csv('dataset/data_bank_customer_survey.csv')
``` 

## 4.Results:




