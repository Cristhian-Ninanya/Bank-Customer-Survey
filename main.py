import utils as utl
import models as mo
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

#================= 1.Data Preprocessing ===========================
#----------------- 1.1 Load Data  -------------------------------
    #Load and print Data set:
    df = utl.load_from_csv('dataset/data_bank_customer_survey.csv')
    
    #Create a copy:
    df_bank = df.copy(deep=True)
    print('\nDataset dimensions: ',df_bank.shape)
    print('\nDataset columns:\n',df_bank.columns)
    print('\nDataSet:\n',df.head(5))

#----------------- 1.2 Dataset Balancing -------------------------------
    #balance dataset:
    df_bank_bal = utl.balance_dataset(df_bank,'y')
    print('\nDataset Balanced Dimensions:',df_bank_bal.shape)

#----------------- 1.3 Split dataset: X (data) and y (target)---------
    #Split unbalanced and balanced dataset:
    X_unb, y_unb = utl.features_target(df_bank,['y'],'y')
    X_bal, y_bal = utl.features_target(df_bank_bal,['y'],'y')

#----------------- 1.4 Convert categorical to numeric and scalar data-----------
    #Generate Colum transformer for data X Columns:
    col_trans = utl.categorical_to_numeric(X_unb)

#----------------- 1.5 Create train and text data ---------------

    X_train_unb, X_test_unb, y_train_unb, y_test_unb = utl.train_test(X_unb, y_unb)
    X_train_bal, X_test_bal, y_train_bal, y_test_bal = utl.train_test(X_bal, y_bal)


#================= 2.Random Forest Classifier ===========================
#----------------- 2.1 RandonForest - Unbalanced Data ------------------
    #Random forest using unbalanced data:
    print("\n----------Results for Random Forest (Unbalanced Data)----------\n")
    metrics_rf_default = mo.random_forest_default(X_train_unb, X_test_unb, y_train_unb, y_test_unb, col_trans)

#----------------- 2.2 Cross Validation - balanced Data -------------------
    #Random forest using balanced data and cross validation:
    print("\n----------Results for Random Forest (Balanced Data)----------\n")
    rf_best_param, metrics_rf_best_param = mo.random_forest_cv(X_train_bal, X_test_bal, y_train_bal, y_test_bal,col_trans)


#================= 3.LogisticRegression classifier ===========================
#----------------- 3.1 Logistic Regression - Unbalanced Data ------------------
    #Logistic Regression using unbalanced data:
    print("\n----------Results for Logistic Regression (Unbalanced Data)----------\n")
    metrics_lr_default = mo.logistic_regression_default(X_train_unb, X_test_unb, y_train_unb, y_test_unb, col_trans)

#----------------- 2.2 Cross Validation - balanced Data -------------------
    #Random forest using balanced data and cross validation:
    print("\n----------Results for Logistic Regression (Balanced Data)----------\n")
    lr_best_param, metrics_lr_best_param = mo.logistic_regression_cv(X_train_bal, X_test_bal, y_train_bal, y_test_bal,col_trans)

#================= 4.SVM classifier ===========================
#----------------- 4.1 SVM - Unbalanced Data ------------------
    #SVM using unbalanced data:
    print("\n----------Results for SVM (Unbalanced Data)----------\n")
    metrics_svm_default = mo.svm_default(X_train_unb, X_test_unb, y_train_unb, y_test_unb, col_trans)

#----------------- 4.2 Cross Validation - balanced Data -------------------
    #SVM using balanced data and cross validation:
    print("\n----------Results for SVM (Balanced Data)----------\n")
    svm_best_param, metrics_svm_best_param = mo.svm_cv(X_train_bal, X_test_bal, y_train_bal, y_test_bal,col_trans)

#================= 5.best classifier selection ================
    #Metrics withous balanced dataset:
    metrics_unb = [metrics_rf_default[1], metrics_lr_default[1],metrics_svm_default[1]]
    df_metrics_unb = pd.DataFrame.from_dict(metrics_unb)

    #Metrics with balanced dataset:
    metrics_bal = [metrics_rf_best_param[1], metrics_lr_best_param[1], metrics_svm_best_param[1]]
    df_metrics_bal = pd.DataFrame.from_dict(metrics_bal)

    #Show results:
    print('\n ----------Classifiers metrics results----------\n')
    print("Unbalanced DataSet Metrics:")
    print(df_metrics_unb)

    figure3 = df_metrics_unb.sort_values(by='recall', ascending=False).plot(x='model', y=['recall', 'accuracy', 'f1_score'],kind='bar')
    plt.title(f"Unbalanced DataSet Metrics")
    plt.savefig('results/figure3.jpg')


 
    print("\n Balanced DataSet Metrics:")
    print(df_metrics_bal)

    figure4 = df_metrics_bal.sort_values(by='recall', ascending=False).plot(x='model', y=['recall', 'accuracy', 'f1_score'],kind='bar')
    plt.title(f"Balanced DataSet Metrics")
    plt.savefig('results/figure4.jpg')

    #Best classifer model:
    dict_best = utl.select_best_classifier(metrics_bal)
    df_best = pd.DataFrame(dict_best)
    print("\n Best Classifier:\n", df_best)
