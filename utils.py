#import libraies:
import pandas as pd
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


#import dataset  (*.csv format):
def load_from_csv(path):
    return pd.read_csv(path, sep = ",")

#Split dataset into 'X'(data) and 'y'(target):
def features_target(dataset, drop_cols, y):
    X = dataset.drop(drop_cols, axis=1)
    y = dataset[y]
    return X, y

#Balance dataset in function of target values proportion:
def balance_dataset(dataset, target):
    #Countplot for target values (unbalanced)
    figure1 = sns.countplot(x=dataset[target], palette='Set2')
    plt.title(f"Umbalanced Taget '{target}'")
    plt.savefig('results/figure1.jpg')

    # split dataframe for each value of dataset (0 and 1):
    df_target_0 = dataset[dataset[target]==0]
    df_target_1 = dataset[dataset[target]==1]

    # Create balanced dataframe for dataset_val_0 (reference dataset_val_1)
    df_sample_0 = df_target_0.sample(df_target_1.shape[0])

    # Create a copy from y_1:
    df_sample_1 = df_target_1.copy(deep=True)

    # Join both balanced dataframe:
    df_balanced = pd.concat([df_sample_0, df_sample_1]) 

    print('\nColumns dataset: \n',df_balanced.columns)
    print('\nBalanced values: \n',df_balanced[target].value_counts())

    #Countplot for target values (unbalanced)
    figure2 = sns.countplot(x=df_balanced[target], palette='Set2')
    plt.title(f"Balanced Target '{target}'")
    plt.savefig('results/figure2.jpg')

    return df_balanced


#Converte categorical to numerical and apply scaler:
def categorical_to_numeric(dataset):
    # Split Dataframe 'X_unbal' and 'X_bal into categorical and numeric:
    categoric_data = dataset.select_dtypes(include='object').columns
    numeric_data = dataset.select_dtypes(include='int64').columns

    # Transform data from categorical to numeric:
    # 1. For categorical data use OneHotEncoder() with dataframe 'categorical_data'
    # 2. For numeric data use StandardScaler()  with dataframe 'numeric_data'
    transformer_data = [('cat', OneHotEncoder(), categoric_data), ('num', StandardScaler(), numeric_data)]
    colum_transform = ColumnTransformer(transformers=transformer_data)

    return colum_transform

#Split 'X' and 'y' into "train" and "test":
def train_test(df_X, df_y):
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.2, random_state=42)

    return X_train, X_test, y_train, y_test

#Parameter list for each classification model:
def Change_List_Parameters(list):
    lst_best = []
    #Loop for parameter:
    for best_param in list.items():
        Parameter = best_param[0].replace(best_param[0].split('__')[0] +'__','') # split parameter name
        lst_best.append(Parameter)       # parameter name
        lst_best.append(best_param[1])  # parameter value

        # Converts parameters list and values into a dictionary
        it_val = iter(lst_best)
        res_dct = dict(zip(it_val, it_val))
    return res_dct

#Selec best classificacion parameters:        
def select_best_classifier(metrics):
    List = sorted(metrics, key = lambda i: i['recall'], reverse = True)[0]
    best_model = List.get('model')
    best_recall = List.get('recall')
    best_f1_score = List.get('f1_score')
    best_acc = List.get('accuracy')
    return [{'model': best_model, 'recall': best_recall, 'f1_score': best_f1_score, 'accuracy': best_acc}]