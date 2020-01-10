#dataset: collection données RNA-seq, extraction random d'expressions
#de gènes de patients ayant des tumeurs differentes: BRCA; KIRC, COAD, LUAD, PRAD

# les samples/ instances stockés row-wise. 

#dummy-name, gene_XX donné à chaque attribut

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def SVM(X, Y):
	training_accuracy, testing_accuracy = [], []
	
	# Search for the best number of feature to get the best accuracy
	for i in range(1, len(X.iloc[1,:])):
	    # Feature selection
	    #Select Kbest: Select features according to the k highest scores.
	    fts = SelectKBest(f_classif, k=i).fit(X, Y)
	    # Obtention of a numpy.ndarray: contains bool like positions of the selected columns 
	    support = fts.get_support()  
	    # Feature filter 
	    new_X = fts.transform(X)
	
	    #Split training and testing data
	    X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33,                      random_state=12)
	
	    #Standardize features by removing the mean and scaling to unit variance
	    #Standardization of a dataset is a common requirement for many machine learning         estimators: they might behave badly if the individual features do not more or less look like    standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
	    sc = StandardScaler()
	    sc.fit(X_train)
	    X_train = sc.transform(X_train)
	    X_test = sc.transform(X_test)
	
	    #Create a simple model
	    # KNeighborsClassifier created 
	    model = svm.SVC(C=0.5, kernel='linear')
	
	    #Train the model
	    model.fit(X_train, Y_train)
	   
	
	    # Teste l'accuracy du modèle
	    training_accuracy.append(accuracy_score(model.predict(X_train),Y_train))
	    testing_accuracy.append(accuracy_score(model.predict(X_test),Y_test))
	    Z = model.predict(X_test)
	
	    # Affiche cette accuracy
	    ("training accuracy:", training_accuracy[i-1], "Testing accuracy:",         testing_accuracy[i-1])
	    print("crosstab:\n", pd.crosstab(Y_test, Z))
	    
	    if i == 15:
	        break
	
	#Plotting the accuracy of the testing and training 
	fig, ax = plt.subplots()
	ax.set_xlabel('Features')
	ax.set_ylabel('Training and Testing accuracy')
	ax.plot(range(len(training_accuracy)), training_accuracy, color="tab:red", label="Training accuracy")
	ax.plot(range(len(training_accuracy)), testing_accuracy, color="tab:green", label="Testing accuracy")
	ax.set_xticks(np.arange(0, len(training_accuracy), 1))
	ax.set_yticks(np.arange(0.5, 1.005, 0.05))
	plt.title("Learning curve")
	plt.grid(True)
	plt.legend()
	plt.show()
	
	
