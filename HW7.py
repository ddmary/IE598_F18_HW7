#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 18:31:34 2018

@author: dongdongmary
"""
#import all packages
import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

#loading dataset

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

#train test split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

#random forest
estimators = []
in_sample = []
out_of_sample= []
for n in range(100, 500, 50):
    forest = RandomForestClassifier(criterion='gini',
                                    n_estimators=n,
                                    random_state=1,
                                    max_depth=4)
    forest.fit(X_train, y_train)
    y_pred_train = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)    
    
    cv_scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    out_of_sample_accuracy =accuracy_score(y_pred_test, y_test)
    
    
    print('Cross Validation scores:',cv_scores.tolist())
    print()
    print('Mean of CV scores:',cv_scores.mean())
    print()
    print('Std of CV scores:',cv_scores.std())
    print()
    print('Out of sample accuracy:', out_of_sample_accuracy)
    print()
    
    estimators.append(n)
    in_sample.append(cv_scores.mean())
    out_of_sample.append(out_of_sample_accuracy)
#make a dataframe
forest_scores = pd.DataFrame({'n_estimators':estimators,
                       'in_sample_accuracy':in_sample,
                       'out_of_sample_accuracy':out_of_sample_accuracy})
    

#output the importance of each feature
forest = RandomForestClassifier(criterion = 'gini',
                                max_depth = 4,
                                n_estimators = 350, 
                                random_state = 1)

feat_labels = df_wine.columns[1:]


forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X_train.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
#plt.savefig('images/04_09.png', dpi=300)
plt.show()



    
    
    
    
#
print('--------------------------------------------------------------------')
print("My name is Yuezhi Li")
print("My NetID is: yuezhi2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
