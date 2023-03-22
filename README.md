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
```terminal
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

* Relative Path: dataset/data_bank_customer_survey.csv (recommended)
```python
#================= 1.Data Preprocessing ===========================
#----------------- 1.1 Load Data  -------------------------------
    #Load and print Data set:
    df = utl.load_from_csv('dataset/data_bank_customer_survey.csv')
``` 
* General Path: C:\ML_scikitlearn\entorno\dataset\data_bank_customer_survey.csv
```python
#================= 1.Data Preprocessing ===========================
#----------------- 1.1 Load Data  -------------------------------
    #Load and print Data set:
    df = utl.load_from_csv('C:/ML_scikitlearn/entorno/dataset/data_bank_customer_survey.csv')
``` 
## 4.Dataset Balancing

* Before Balancing

![alt text](https://github.com/Cristhian-Ninanya/Bank-Customer-Survey/blob/master/results/figure1.jpg?raw=true)

**OBS:**
* Dataset is unbalanced for 'y' (target). It is necessary to balance dataset to avoid overtraining.

* 2.After Balancing

![alt text](https://github.com/Cristhian-Ninanya/Bank-Customer-Survey/blob/master/results/figure2.jpg?raw=true)


## 5.Training results:

* **Random Forest**

```terminal
*********** Results for Random Forest (Unbalanced Data) ********
------------------------------------------------------
*f1_score results for RandomForestClassifier: 50.761%
*Accuracy for RandomForestClassifier: 65.982%
*Recall for  RandomForestClassifier : 41.247%
-----------------------------------------------------
Training duration: 0:00:05.458296 seconds.

*********** Results for Random Forest (Balanced Data) ********
-----------------------------------------
Training time Cross Validation Random Forest
Time: 0:00:23.628573 seconds.
-----------------------------------------
Best parameters for RandomForest:
 {'criterion': 'entropy', 'max_depth': 2, 'max_features': 'log2', 'n_estimators': 200}
--------------------------------------------------------------------------------------
*f1_score results for RandomForestClassifier: 76.620%
*Accuracy for RandomForestClassifier: 75.354%
*Recall for  RandomForestClassifier : 77.930%
---------------------------------------------
Training duration: 0:00:00.670183 seconds.
```
* **Logistic Regression**

```terminal
********* Results for Logistic Regression (Unbalanced Data) ***********
----------------------------------------------------
*f1_score results for LogisticRegression: 44.846%
*Accuracy for LogisticRegression: 65.493%
*Recall for  LogisticRegression : 34.097%
----------------------------------------------------
Training duration: 0:00:00.692418 seconds.


*********** Results for Logistic Regression (Balanced Data) ***********
-----------------------------------------
Training time Cross Validation Logistic Regression:
Time: 0:00:30.122865 seconds.
-----------------------------------------
Best parameters for LogisticRegression:
 {'C': 1.0, 'l1_ratio': 0.5, 'max_iter': 5000, 'penalty': 'elasticnet', 'solver': 'saga', 'tol': 0.005}
----------------------------------------------------------------------
*f1_score results for LogisticRegression: 82.107%
*Accuracy for LogisticRegression: 83.603%
*Recall for  LogisticRegression : 80.664%
----------------------------------------------------
Training duration: 0:00:00.357222 seconds.

```
* **Support Vector Machines**

```terminal
*********** Results for SVM (Unbalanced Data) ***********
----------------------------------------------------
*f1_score results for SVC: 46.924%
*Accuracy for SVC: 68.607%
*Recall for  SVC : 35.655%
----------------------------------------------------
Training duration: 0:00:37.845822 seconds.


*********** Results for SVM (Balanced Data) ***********
-----------------------------------------
Training time Cross Validation SVM:
Time: 0:03:00.805451 seconds.
-----------------------------------------
Best parameters for LogisticRegression:
 {'C': 10, 'kernel': 'linear'}
------------------------------------------
*f1_score results for SVC: 82.877%
*Accuracy for SVC: 83.039%
*Recall for  SVC : 82.715%
-------------------------------------------
Training duration: 0:00:10.859407 seconds.

```

## 5.Final Results and observations:

```terminal
 ----------Classifiers metrics results----------

Unbalanced DataSet Metrics:
                    model  f1_score  accuracy    recall
0  RandomForestClassifier  0.507614  0.659824  0.412466
1      LogisticRegression  0.448463  0.654930  0.340972
2                     SVC  0.469240  0.686067  0.356554

 Balanced DataSet Metrics:
                    model  f1_score  accuracy    recall
0  RandomForestClassifier  0.766203  0.753541  0.779297
1      LogisticRegression  0.821074  0.836032  0.806641
2                     SVC  0.828767  0.830392  0.827148

 Best Classifier:
   model    recall  f1_score  accuracy
0   SVC  0.827148  0.828767  0.830392
```

* Unbalanced DataSet Metrics:

![alt text](https://github.com/Cristhian-Ninanya/Bank-Customer-Survey/blob/master/results/figure3.jpg?raw=true)

* Balanced DataSet Metrics:

![alt text](https://github.com/Cristhian-Ninanya/Bank-Customer-Survey/blob/master/results/figure4.jpg?raw=true)

* **observations**

    * Accuracy is not enough to define the best classifier model.
    * Recall has better results to chose the best classifier model.
    * After balancing the DataSet all classifier models have better results.
    * Best classifier is "SVC" (0.83 recall), however "RandomForest" is much faster (1.68 seg). 
    * We can improve RandomForest results (increase ```max_depth```), but it can get closer to overfitting.



