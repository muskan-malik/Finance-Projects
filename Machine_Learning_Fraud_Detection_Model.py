#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 09:49:36 2024

@author: muskan
"""




############################################################################################################################

# # Code CART and Random Forest Model with K-Fold cross Validation 

############################################################################################################################

# Following code is divided into two parts ( k-fold and train test).
#please add an apostrophe below to run the second part and make K-fold part a comment.

''
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the training data
train_data = pd.read_csv('card_transdata_train.csv')

# Split features and target variable
X_train = train_data.drop(columns=['fraud'])
y_train = train_data['fraud']

# Load the test data
test_data = pd.read_csv('card_transdata_test.csv')
X_test = test_data.drop(columns=['fraud'])
y_test = test_data['fraud']

# Function to print and return confusion matrix
def get_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(title)
    print(cm)
    print(f'True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}, False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}')
    return cm

# K-Fold Cross-Validation
def k_fold_evaluation(X, y, k, classifier, classifier_name):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []
    cms = []

    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X.iloc[train_index], X.iloc[test_index]
        y_train_k, y_test_k = y.iloc[train_index], y.iloc[test_index]
        
        classifier.fit(X_train_k, y_train_k)
        y_pred = classifier.predict(X_test_k)
        
        accuracies.append(accuracy_score(y_test_k, y_pred))
        precisions.append(precision_score(y_test_k, y_pred))
        recalls.append(recall_score(y_test_k, y_pred))
        f1s.append(f1_score(y_test_k, y_pred))
        aucs.append(roc_auc_score(y_test_k, y_pred))
        cms.append(confusion_matrix(y_test_k, y_pred))

    print(f"{classifier_name} Performance with {k}-Fold Cross-Validation:")
    print(f'Average Accuracy: {sum(accuracies) / k}')
    print(f'Average Precision: {sum(precisions) / k}')
    print(f'Average Recall: {sum(recalls) / k}')
    print(f'Average F1 Score: {sum(f1s) / k}')
    print(f'Average AUC: {sum(aucs) / k}')
    print()
    
    return cms

# Evaluate on provided test data
def evaluate_on_test_data(X_train, X_test, y_train, y_test, classifier, classifier_name):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"{classifier_name} Performance on Test Data:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')
    print()
    
    return get_confusion_matrix(y_test, y_pred, title=f'Confusion Matrix for {classifier_name} (Test Data)')

# 3-Fold Cross-Validation
#cms_dt_3_fold = k_fold_evaluation(X_train, y_train, k=3, classifier=DecisionTreeClassifier(random_state=42, max_depth=10), classifier_name="Decision Tree")
#cms_rf_3_fold = k_fold_evaluation(X_train, y_train, k=3, classifier=RandomForestClassifier(random_state=42, max_depth=10), classifier_name="Random Forest")

# 5-Fold Cross-Validation
#cms_dt_5_fold = k_fold_evaluation(X_train, y_train, k=5, classifier=DecisionTreeClassifier(random_state=42, max_depth=10), classifier_name="Decision Tree")
#cms_rf_5_fold = k_fold_evaluation(X_train, y_train, k=5, classifier=RandomForestClassifier(random_state=42, max_depth=10), classifier_name="Random Forest")

# 10-Fold Cross-Validation
cms_dt_10_fold = k_fold_evaluation(X_train, y_train, k=10, classifier=DecisionTreeClassifier(random_state=42, max_depth=10), classifier_name="Decision Tree")
cms_rf_10_fold = k_fold_evaluation(X_train, y_train, k=10, classifier=RandomForestClassifier(random_state=42, max_depth=10), classifier_name="Random Forest")

# Evaluate on provided test data
print("Evaluating on provided test data:")
cm_dt_test = evaluate_on_test_data(X_train, X_test, y_train, y_test, DecisionTreeClassifier(random_state=42, max_depth=10), "Decision Tree")
cm_rf_test = evaluate_on_test_data(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=42, max_depth=10), "Random Forest")

############################################################################################################################

#  Random Forest Model testing on OOS1, OOS2 and OOS3 data files

############################################################################################################################

#_____________________________________________________________________________________________________________________________________#
# OOS1
# Function to test new data
def test_new_data(new_data_path, classifier):
    new_data = pd.read_csv("OOS1.csv")
    X_new = new_data.drop(columns=['fraud'])
    y_new = new_data['fraud']
    return evaluate_on_test_data(X_train, X_new, y_train, y_new, classifier, classifier.__class__.__name__)

# Example usage to test new data
cm_new_data_rf = test_new_data('OOS1.csv', RandomForestClassifier(random_state=42, max_depth=10))
#_____________________________________________________________________________________________________________________________________#

# OOS2
# Function to test new data
def test_new_data(new_data_path, classifier):
    new_data = pd.read_csv("OOS2.csv")
    X_new = new_data.drop(columns=['fraud'])
    y_new = new_data['fraud']
    return evaluate_on_test_data(X_train, X_new, y_train, y_new, classifier, classifier.__class__.__name__)

# Example usage to test new data
cm_new_data_rf = test_new_data('OOS2.csv', RandomForestClassifier(random_state=42, max_depth=10))

#____________________________________________________________________________________________________________#

# OOS3
# Function to test new data
def test_new_data(new_data_path, classifier):
    new_data = pd.read_csv("OOS3.csv")
    X_new = new_data.drop(columns=['fraud'])
    y_new = new_data['fraud']
    return evaluate_on_test_data(X_train, X_new, y_train, y_new, classifier, classifier.__class__.__name__)


# Example usage to test new data
cm_new_data_rf = test_new_data('OOS3.csv', RandomForestClassifier(random_state=42, max_depth=10))

#_____________________________________________________________________________________________________________________________________#

''
'''
############################################################################################################################

# Code CART and Random Forest Model with train-test data partitioning 

############################################################################################################################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the data
train_data = pd.read_csv('card_transdata_train.csv')
test_data = pd.read_csv('card_transdata_test.csv')

# Split features and target variable
X_train = train_data.drop(columns=['fraud'])
y_train = train_data['fraud']
X_test = test_data.drop(columns=['fraud'])
y_test = test_data['fraud']

# Function to print and return confusion matrix
def get_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    print(title)
    print(cm)
    print(f'True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}, False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}')
    return cm

# Function to train and evaluate classifier
def classifier_evaluation(X_train, X_test, y_train, y_test, classifier, classifier_name, split_name):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = get_confusion_matrix(y_test, y_pred, title=f'Confusion Matrix for {classifier_name} ({split_name})')
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    print(f"{classifier_name} ({split_name}) Performance:")
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'AUC: {auc}')
    print()
    
    return cm

# 80/20 Split
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
cm_dt_80_20 = classifier_evaluation(X_train_split, X_test_split, y_train_split, y_test_split, DecisionTreeClassifier(random_state=42, max_depth=10), "Decision Tree", "80/20 split")
cm_rf_80_20 = classifier_evaluation(X_train_split, X_test_split, y_train_split, y_test_split, RandomForestClassifier(random_state=42, max_depth=10), "Random Forest", "80/20 split")

# 60/40 Split
#X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
#cm_dt_60_40 = classifier_evaluation(X_train_split, X_test_split, y_train_split, y_test_split, DecisionTreeClassifier(random_state=42, max_depth=10), "Decision Tree", "60/40 split")
#cm_rf_60_40 = classifier_evaluation(X_train_split, X_test_split, y_train_split, y_test_split, RandomForestClassifier(random_state=42, max_depth=10), "Random Forest", "60/40 split")

# Evaluate on provided test data
print("Evaluating on provided test data:")
cm_dt_test = classifier_evaluation(X_train, X_test, y_train, y_test, DecisionTreeClassifier(random_state=42, max_depth=10), "Decision Tree", "Test Data")
cm_rf_test = classifier_evaluation(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=42, max_depth=10), "Random Forest", "Test Data")

#_____________________________________________________________________________________________________________________________________#

# Function to test new data
def test_new_data(new_data_path, classifier):
    new_data = pd.read_csv(new_data_path)
    X_new = new_data.drop(columns=['fraud'])
    y_new = new_data['fraud']
    cm_new = classifier_evaluation(X_train, X_new, y_train, y_new, classifier, classifier.__class__.__name__, "New Data")
    return cm_new

# Example usage to test new data
cm_new_data_dt = test_new_data('/mnt/data/new_test_data.csv', DecisionTreeClassifier(random_state=42, max_depth=10))
cm_new_data_rf = test_new_data('/mnt/data/new_test_data.csv', RandomForestClassifier(random_state=42, max_depth=10))

#_____________________________________________________________________________________________________________________________________#

'''




############################################################################################################################
