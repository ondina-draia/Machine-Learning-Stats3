#!/usr/bin/python3
# -*- coding: utf-8 -*-

#dataset: collection données RNA-seq, extraction random d'expressions
#de gènes de patients ayant des tumeurs differentes: BRCA; KIRC, COAD, LUAD, PRAD

# les samples/ instances stockés row-wise. 

#dummy-name, gene_XX donné à chaque attribut
import sys
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_networks import kerasFunct


def projectMain(data, label):
	"""
	data : Dataset contenant les données sur les tumeurs
	label : Dataset contenant les noms de chaque tumeur associées aux genes
	
	"""
	#stock features inside X and classes in Y
	df = pd.read_csv(data, low_memory=False) #ficher source des données
	X = df.iloc[:,2:]
	dl = pd.read_csv(label)
	Y = dl.iloc[:,1]
	
	# Feature selection
	fts = SelectKBest(f_classif, k=12).fit(X, Y)
	# Obtention of a numpy.ndarray: contains bool like positions of the selected columns 
	support = fts.get_support()  
	# Feature filter 
	new_X = fts.transform(X)
	
	#Split training and testing data
	X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33, random_state=42)
	
	#Standardize features by removing the mean and scaling to unit variance
	#Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual features do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
	sc = StandardScaler()
	sc.fit(X_train)
	X_train = sc.transform(X_train)
	X_test = sc.transform(X_test)
	
	print(linearModel(X_train, X_test, Y_train, Y_test))
	print(kerasFunct(X_train, X_test, Y_train, Y_test))
	
def linearModel(X_train, X_test, Y_train, Y_test):
	"""
	X_train :
	X_test :
	Y_train :
	Y_test :
	
	"""
	#Create a simple model
	logreg = linear_model.LogisticRegression()
	
	#Train the model
	logreg.fit(X_train,Y_train)
	Z = logreg.predict(X_test)
	
	# Teste l'accuracy du modèle
	training_accuracy = accuracy_score(logreg.predict(X_train),Y_train)
	testing_accuracy = accuracy_score(logreg.predict(X_test),Y_test)
	
	# Affiche cette accuracy
	print("training accuracy:", training_accuracy, "Testing accuracy:", testing_accuracy)
	print("crosstab:\n", pd.crosstab(Y_test, Z))



#En construction
#sns.set_style("whitegrid")
#sns.FacetGrid(df, hue="Class", height=4) \
#    .map(plt.scatter, "Unnamed: 0", "Class") \
#    .add_legend()
#plt.show()


if __name__ == '__main__':
	data = sys.argv[1]
	label = sys.argv[2]
	print(projectMain(data, label))
