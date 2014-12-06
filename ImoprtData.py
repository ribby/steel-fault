# -*- coding: utf-8 -*-
"""
Created on Wed Dec 03 01:16:39 2014

@author: oromi_000
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm

index = range(0,1940)
col = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 
       'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity',
       'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300',
       'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index',
       'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index',
       'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 
       'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas', 'Pastry',
       'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps',
       'Other_Faults']
df = pd.DataFrame.from_csv('steel.txt', sep='\t')

# Replaces X_Min as index with 0...n
df = df.reset_index()

# Break up and convert data
X = df[col[0:27]]
X = X.as_matrix()

y = df[col[28]]
y = y.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                    random_state=123)
                                                    
# Feature scaling
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

#PCA
#sklearn_pca = PCA(n_components=2)
#transf_pca = sklearn_pca.fit_transform(X_train)
#print transf_pca

# DT
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
print "The DT accuracy is",acc


