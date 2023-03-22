
import utils as utl

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn import svm


def calculate_metrics (model, y_test, y_pred):
  # Calculate Metrics:
  f1 = f1_score(y_test, y_pred)
  accuracy = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)

  #Print results:
  print("----------------------------------------------------------------------")
  print("*f1_score results for " + type(model).__name__ + ": {0:.3%}".format(f1))
  print("*Accuracy for " + type(model).__name__ + ": {0:.3%}".format(accuracy))
  print("*Recall for  " + type(model).__name__ + " : {0:.3%}".format(recall))

  return f1, accuracy, recall

 
def train_predict_model(model, X_train, X_test, y_train, y_test, col_trans):
  
  #Apply model, calculate training time and metrics:
  start = datetime.now()
  pipe_model = make_pipeline(col_trans, model)
  pipe_model.fit(X_train, y_train)
  y_pred = pipe_model.predict(X_test)

  #Calculate metrics:
  f1, accuracy, recall = calculate_metrics(model, y_test, y_pred)
  end = datetime.now()

  #Print results:
  print('----------------------------------------------------')
  print('Training duration: {} segundos.'.format(end - start))
  
  return pipe_model,  {'model': type(model).__name__, 'f1_score': f1, 'accuracy': accuracy, 'recall': recall}


def random_forest_default(X_train, X_test, y_train, y_test, colum_transform):

  #Create model and apply dedault parameter:
  rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
  #Train and predict model:
  metrics_rf_default = train_predict_model(rf_default, X_train, X_test, y_train, y_test, colum_transform)
  
  return metrics_rf_default



def random_forest_cv(X_train, X_test, y_train, y_test, colum_transform):

    #Create model for cross validation:
    rf_cv = RandomForestClassifier(random_state=42)
  
    model_rf_cv = make_pipeline(colum_transform, rf_cv)

    #Create parameter grid:
    param_gridcv_rf = {
        'randomforestclassifier__n_estimators': [200],
        'randomforestclassifier__max_features': ['sqrt', 'log2'],
        'randomforestclassifier__max_depth' : [2],
        'randomforestclassifier__criterion' : ['gini' , 'entropy']
    }

    #Create cross validation for Random Forest:
    GS_rf_cv = GridSearchCV(model_rf_cv, param_gridcv_rf, cv=10)

    #Training Classifier:
    start = datetime.now()
    GS_rf_cv.fit(X_train, y_train)
    end = datetime.now()
    print('-----------------------------------------')
    print('Training time Cross Validation Random Forest')
    print('Time: {} seconds.'.format(end - start))

    #best parameter for random fores:
    param_rf = utl.Change_List_Parameters(GS_rf_cv.best_params_)
    print('-----------------------------------------')
    print('Best parameters for RandomForest:\n',param_rf)

    # RF model with best parameter:
    rf_best_param = RandomForestClassifier(**param_rf, random_state=42)

    #Calculate Metrics:
    metrics_rf_best_param = train_predict_model(rf_best_param, X_train, X_test, y_train, y_test, colum_transform)
    
    return rf_best_param, metrics_rf_best_param


def logistic_regression_default(X_train, X_test, y_train, y_test, colum_transform):

  #Create model and apply dedault parameter:
  lr_default = LogisticRegression(random_state=42, max_iter=500)
  #Train and predict model:
  metrics_lr_default = train_predict_model(lr_default, X_train, X_test, y_train, y_test, colum_transform)
  
  return metrics_lr_default


def logistic_regression_cv(X_train, X_test, y_train, y_test, colum_transform):

    #Create model for cross validation:
    lr_cv = LogisticRegression(random_state=42)
    model_lr_cv = make_pipeline(colum_transform, lr_cv)

    #Create parameter grid:
    param_gridcv_lr = [
        {
            'logisticregression__penalty': ['l1'],
            'logisticregression__C': [1.0, 5.0],     
            'logisticregression__solver': ['liblinear', 'saga'],
            'logisticregression__max_iter' : [2500],
            'logisticregression__tol' : [0.005]
        },
        {
            'logisticregression__penalty': ['l2'],
            'logisticregression__C': [1.0, 5.0],     
            'logisticregression__solver': ['lbfgs','newton-cg'],
            'logisticregression__max_iter' : [5000],
            'logisticregression__tol' : [0.005]
        },
        {
          'logisticregression__penalty': ['elasticnet'],
          'logisticregression__C': [1.0, 5.0],
          'logisticregression__solver': ['saga'],
          'logisticregression__max_iter' : [5000],
          'logisticregression__l1_ratio' : [0.1, 0.5, 0.9],
          'logisticregression__tol' : [0.005]
        }
    ]

    #Create cross validation for Logistic Regression:
    GS_lr_cv = GridSearchCV(model_lr_cv, param_gridcv_lr, cv=10)

    #Training Classifier:
    start = datetime.now()
    GS_lr_cv.fit(X_train, y_train)
    end = datetime.now()
    print('-----------------------------------------')
    print('Training time Cross Validation Logistic Regression:')
    print('Time: {} seconds.'.format(end - start))

    #best parameter for random forest:
    param_lr = utl.Change_List_Parameters(GS_lr_cv.best_params_)
    print('-----------------------------------------')
    print('Best parameters for LogisticRegression:\n',param_lr)

    # LR model with best parameter:
    lr_best_param = LogisticRegression(**param_lr, random_state=42)

    #Calculate Metrics:
    metrics_lr_best_param = train_predict_model(lr_best_param, X_train, X_test, y_train, y_test, colum_transform)
    
    return lr_best_param, metrics_lr_best_param
  
def svm_default(X_train, X_test, y_train, y_test, colum_transform):

  #Create model and apply dedault parameter:
  svm_default = svm.SVC(random_state=42)
  #Train and predict model:
  metrics_svm_default = train_predict_model(svm_default, X_train, X_test, y_train, y_test, colum_transform)
  
  return metrics_svm_default


def svm_cv(X_train, X_test, y_train, y_test, colum_transform):

    #Create model for cross validation:
    svm_cv = svm.SVC(random_state=42)
    model_svm_cv = make_pipeline(colum_transform, svm_cv)

    #Create parameter grid:
    param_gridcv_smv =  [
      {
        'svc__kernel': ['rbf'],
        'svc__gamma': [0.01],
        'svc__C': [0.1],
        'svc__tol': [0.005]
      },
      {
        'svc__kernel': ['sigmoid'],
        'svc__gamma': [1],
        'svc__C': [0.01],
        'svc__tol': [0.001]
      },
      {
        'svc__kernel': ['linear'], 
        'svc__C': [10]
      }
    ] 

    #Create cross validation for svm:
    GS_svm_cv = GridSearchCV(model_svm_cv, param_gridcv_smv, cv=10)

    #Training Classifier:
    start = datetime.now()
    GS_svm_cv.fit(X_train, y_train)
    end = datetime.now()
    print('-----------------------------------------')
    print('Training time Cross Validation SVM:')
    print('Time: {} seconds.'.format(end - start))

    #best parameter for random fores:
    param_svm = utl.Change_List_Parameters(GS_svm_cv.best_params_)
    print('-----------------------------------------')
    print('Best parameters for LogisticRegression:\n',param_svm)

    # SVM model with best parameter:
    svm_best_param = svm.SVC(**param_svm, random_state=42)

    #Calculate Metrics:
    metrics_svm_best_param = train_predict_model(svm_best_param, X_train, X_test, y_train, y_test, colum_transform)
    
    return svm_best_param, metrics_svm_best_param