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

def tree(X_train, X_test, Y_train, Y_test):
	"""
	
	"""
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
	
