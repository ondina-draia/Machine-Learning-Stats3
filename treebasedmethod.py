# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:54:47 2020

@author: Draia-Nicolau Ondina & Baheux Melina
"""

#Tree based methods
#dataset: collection données RNA-seq, extraction random d'expressions
#de gènes de patients ayant des tumeurs differentes: BRCA; KIRC, COAD, LUAD, PRAD

# les samples/ instances stockés row-wise. 

#dummy-name, gene_XX donné à chaque attribut

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn import tree
from os import system

#stock features inside X and classes in Y
df = pd.read_csv('data.csv', low_memory=False) #ficher source des données
X = df.iloc[:,2:]
dl = pd.read_csv('labels.csv')
Y = dl.iloc[:,1]

# Feature selection
fts = SelectKBest(f_classif, k=12).fit(X, Y)
# Obtention of a numpy.ndarray: contains bool like positions of the selected columns 
support = fts.get_support()  
# Feature filter 
new_X = fts.transform(X)
# Divide the data in 67% training data, 33% test
X_train, X_test, Y_train, Y_test = train_test_split(new_X, Y, test_size=0.33, random_state=13)

#Standardize features by removing the mean and scaling to unit variance
#Standardization of a dataset is a common requirement for many machine learning         estimators: they might behave badly if the individual features do not more or less look like    standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, Y_train)

label_names = np.array(['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'])

data_names = np.array(["Gene_"+str(i) for i in range(new_X.shape[1])])


estimator = model.estimators_[5]
dotfile = open("C:/Users/utilisateur/Documents/Cours/M2 DLAD Bioinfo/Machine-Learning-Stats3-master/output/tree.dot", 'w')
tree.export_graphviz(estimator, out_file='output/tree.dot', 
                feature_names = data_names,
                class_names = label_names,
                rounded = True, proportion = False,
                precision = 2, filled = True)
dotfile.close()

call(['dot', '-Tpng', "C:/Users/utilisateur/Documents/Cours/M2 DLAD Bioinfo/Machine-Learning-Stats3-master/output/tree.dot", '-o', 'output/tree.png', '-Gdpi=1000']) #recuperer fichier generée sous le nom de tree, copier le contenu et le coller sur http://webgraphviz.com/ afin de el visualiser

print(accuracy_score(model.predict(X_test), Y_test))
