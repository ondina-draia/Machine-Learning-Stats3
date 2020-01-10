#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 14:54:47 2020
@author: Draia-Nicolau Ondina & Baheux Melina
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
from subprocess import call
from sklearn import tree
from os import system
from sklearn import svm
import sys
from keras.layers import Dense, Input
from keras.models import Model

def boucleTraining(X,Y, mod):
    """
    X : List of the genes
    Y : List of labels
    mod : regression or svm model
    return : testing_accuracy, training_accuracy, X_test, X_train, Y_test and Y_train values
    """
    training_accuracy, testing_accuracy = [], []
        # Search for the best number of feature to get the best accuracy
    for i in range(1, len(X.iloc[1,:])):
        # Feature selection
        # Select Kbest: Select features according to the k highest scores.
        fts = SelectKBest(f_classif, k=i).fit(X, Y)
        # Obtention of a numpy.ndarray: contains bool like positions of the selected columns 
        support = fts.get_support()  
        # Feature filter 
        new_X = fts.transform(X)
        
        if mod == "reg":
            #Split training and testing data
            X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33, random_state=42)
            
            #Standardize features by removing the mean and scaling to unit variance
            #Standardization of a dataset is a common requirement for many machine learning         estimators: they might behave badly if the individual features do not more or less look like    standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            #Create a simple model
            logreg = linear_model.LogisticRegression(C = 1e30, solver='liblinear')
            
            #Train the model
            logreg.fit(X_train,Y_train)
            Z = logreg.predict(X_test)
            
            # Teste l'accuracy du modèle
            training_accuracy.append(accuracy_score(logreg.predict(X_train),Y_train))
            testing_accuracy.append(accuracy_score(logreg.predict(X_test),Y_test))
        else:
            #Split training and testing data
            X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33, random_state=12)
            
            #Standardize features by removing the mean and scaling to unit variance
            #Standardization of a dataset is a common requirement for many machine learning         estimators: they might behave badly if the individual features do not more or less look like    standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).
            sc = StandardScaler()
            sc.fit(X_train)
            X_train = sc.transform(X_train)
            X_test = sc.transform(X_test)
            #Create a simple model
            # KNeighborsClassifier created 
            model = svm.SVC(C = 100, kernel='linear')
            
            #Train the model
            model.fit(X_train, Y_train)
            Z = model.predict(X_test)
            
            # Teste l'accuracy du modèle
            training_accuracy.append(accuracy_score(model.predict(X_train),Y_train))
            testing_accuracy.append(accuracy_score(model.predict(X_test),Y_test))
            
            
            
        # Affiche cette accuracy
        ("training accuracy:", training_accuracy[i-1], "Testing accuracy:",         testing_accuracy[i-1])
        #~ print("crosstab:\n", pd.crosstab(Y_test, Z))
        
        if i == 15:
            break
    return testing_accuracy, training_accuracy, X_test, X_train, Y_test, Y_train

def plotLinearRegression(X, Y):
    """
    X : List of the genes
    Y : List of labels
    """
    testing_accuracy, training_accuracy, X_test, X_train, Y_test, Y_train = boucleTraining(X, Y, "reg")
    #~ #Plotting the accuracy of the testing and training 
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


def treeFunc(X_train, X_test, Y_train, Y_test, new_X):
    """
    X_train : Trained values
    X_test : Test values
    Y_train : Labels of trained values
    Y_test : Labels of test values
    new_X : Selection of kbest
    """
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X_train, Y_train)
    
    label_names = np.array(['BRCA', 'COAD', 'KIRC', 'LUAD', 'PRAD'])
    
    data_names = np.array(["Gene_"+str(i) for i in range(new_X.shape[1])])
    
    
    estimator = model.estimators_[5]
    dotfile = open("./tree.dot", 'w')
    tree.export_graphviz(estimator, out_file='./tree.dot', 
                    feature_names = data_names,
                    class_names = label_names,
                    rounded = True, proportion = False,
                    precision = 2, filled = True)
    dotfile.close()



def SVM(X, Y):
    """
    X : List of the genes
    Y : List of labels
    """
    testing_accuracy, training_accuracy, X_test, X_train, Y_test, Y_train = boucleTraining(X, Y, "svm")
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
    
def compTrainTest(X, Y):
    """
    X : List of the genes
    Y : List of labels
    """
    testing_accuracy, training_accuracy, X_test, X_train, Y_test, Y_train = boucleTraining(X, Y, "reg")
    
    noise_factor = 0.5
    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)
        
    #recuperer données dans des listes
    BRCA_test = []
    PRAD_test = []
    LUAD_test = []
    COAD_test = []
    KIRC_test = []
    
    k=0
    LABELS = ['BRCA','PRAD','LUAD','COAD','KIRC']
    
    for j in Y_test:
        l=0
        for i in np.nditer(X_test):
            if l==k:
                if j == 'BRCA':
                    BRCA_test.append(float(i))
                elif j == 'PRAD':
                    PRAD_test.append(float(i))
                elif j == 'LUAD':
                    LUAD_test.append(float(i))
                elif j == 'COAD':
                    COAD_test.append(float(i))
                else:
                    KIRC_test.append(float(i))
            if l > k:
                break 
            l+=1
        k+=1
    
    BRCA_train = []
    PRAD_train = []
    LUAD_train = []
    COAD_train = []
    KIRC_train = []
    
    k=0
    LABELS = ['BRCA','PRAD','LUAD','COAD','KIRC']
    
    for j in Y_train:
        l=0
        for i in np.nditer(X_train):
            if l==k:
                if j == 'BRCA':
                    BRCA_train.append(float(i))
                elif j == 'PRAD':
                    PRAD_train.append(float(i))
                elif j == 'LUAD':
                    LUAD_train.append(float(i))
                elif j == 'COAD':
                    COAD_train.append(float(i))
                else:
                    KIRC_train.append(float(i))
            if l > k:
                break 
            l+=1
        k+=1
    
    plt.figure(figsize=(20, 2))
    for i in range(len(LABELS)):
        ax = plt.subplot(1, len(LABELS), i+1)
        #plt.gray()
        if LABELS[i] == 'BRCA':
            plt.scatter(range(0, len(BRCA_test)),BRCA_test)
        elif LABELS[i] == 'PRAD':
            plt.scatter(range(0, len(PRAD_test)),PRAD_test)   
        elif LABELS[i] == 'LUAD':
            plt.scatter(range(0, len(LUAD_test)),LUAD_test)
        elif LABELS[i] == 'COAD':
            plt.scatter(range(0, len(COAD_test)),COAD_test)  
        elif LABELS[i] == 'KIRC':
            plt.scatter(range(0, len(KIRC_test)),KIRC_test)
            
            
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.title(LABELS[i])
    
    plt.figure(figsize=(20, 2))
    for i in range(len(LABELS)):
        ax = plt.subplot(1, len(LABELS), i+1)
        #plt.gray()
        if LABELS[i] == 'BRCA':
            plt.scatter(range(0, len(BRCA_train)),BRCA_train)
        elif LABELS[i] == 'PRAD':
            plt.scatter(range(0, len(PRAD_train)),PRAD_train)
        elif LABELS[i] == 'LUAD':
            plt.scatter(range(0, len(LUAD_train)),LUAD_train)
        elif LABELS[i] == 'COAD':
            plt.scatter(range(0, len(COAD_train)),COAD_train)
        elif LABELS[i] == 'KIRC':
            plt.scatter(range(0, len(KIRC_train)),KIRC_train)
            
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(True)
        plt.title(LABELS[i])
    
    
    plt.show()



def linearModel(X_train, X_test, Y_train, Y_test):
    """
    X_train : Trained values
    X_test : Test values
    Y_train : Labels of trained values
    Y_test : Labels of test values
    
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

def kerasNN(X, Y):
    """
    X: features
    Y: labels 
    This function creates a neural network model using keras and trains the model using the data from data.csv
    """
    #stock features inside X and classes in Y
    #df = pd.read_csv(data, low_memory=False) #ficher source des données
    #X = df.iloc[:,2:]
    #dl = pd.read_csv(label)
    #Y = dl.iloc[:,1]
    # (801, 20530) so 801 features and 20530 examples

    #we need to encode our data from 0 to 4 corresponding to each cancer type
    Y_encoded = list()
    for i in Y :
        if i == 'PRAD' : Y_encoded.append(0)
        if i == 'LUAD' : Y_encoded.append(1)
        if i == 'BRCA' : Y_encoded.append(2)
        if i == 'KIRC' : Y_encoded.append(3)
        if i == 'COAD' : Y_encoded.append(4)
    
    from keras.utils import to_categorical
    Y_categorical = to_categorical(Y_encoded) #create a categorical type of Y for the labels


    # Split data into testing and training sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_categorical, test_size=0.33, random_state=42)

    # Create the structure API : specify the preceding layer
    init = 'random_uniform'
    input_layer = Input(shape=(20530,))
    mid_layer = Dense(15, activation = 'relu', kernel_initializer = init)(input_layer)
    mid_layer_2 = Dense(8, activation = 'relu', kernel_initializer = init)(mid_layer)
    output_layer = Dense(5, activation = 'softmax', kernel_initializer = init)(mid_layer_2)

    # Compile the model
    model = Model(input = input_layer, output = output_layer)
    model.compile(optimizer='sgd',loss='binary_crossentropy',metrics=['accuracy'])
    
    model.summary()

    # Fitting the model
    model.fit(X_train, Y_train, batch_size=32, epochs=20, verbose=1)

    # Predict with the model
    Z = model.predict(X_test)

    # Compare the prediction with the test data
    Z_vect = np.argmax(Z, axis=1)
    Y_test_vect = np.argmax(Y_test, axis=1)
    #argmax: Returns the indices of the maximum values along an axis.
    Comp = pd.crosstab(Y_test_vect, Z_vect) # compare the predictions and the test data as vectors
    print(Comp)

    # Display a learning curve
    # dataframe with the training and testing accuracies
    dfAccuracy = pd.DataFrame(columns=["training_accuracy", "testing_accuracy"])

    for i in range(10,len(X_train.index)): 
        model.fit(X_train.iloc[1:i,],Y_train[1:i], batch_size = 64, epochs = 20) # don't train too much a neural network 

        # Convert everything we need into vectors thanks to argmax(), cause sklearn functions work with vectors
        ytrain = np.argmax(Y_train, axis=1)
        predtrain = np.argmax(model.predict(X_train), axis=1)
        ytest = np.argmax(Y_test, axis=1)
        predtest = np.argmax(model.predict(X_test), axis=1)

        training_accuracy = accuracy_score(ytrain, predtrain)
        testing_accuracy = accuracy_score(ytest, predtest)
        dfAccuracy = dfAccuracy.append({"training_accuracy": training_accuracy, "testing_accuracy": testing_accuracy}, ignore_index=True)
        print(dfAccuracy)

    t = range(len(X_train.index)-10)
    a = dfAccuracy.iloc[:,0] # training accuracies
    b = dfAccuracy.iloc[:,1] # testing accuracies

    plt.plot(t, a, 'r')
    plt.plot(t, b, 'g')
    plt.title("Learning curve")
    plt.xlabel('Features')
    plt.ylabel('Training and Testing accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = sys.argv[1]
    label = sys.argv[2]
    
    #stock features inside X and classes in Y
    df = pd.read_csv(data, low_memory=False) #ficher source des données
    X = df.iloc[:,2:]
    dl = pd.read_csv(label)
    Y = dl.iloc[:,1]
    #~ print(boucleTraining(X,Y))
    print(plotLinearRegression(X, Y))
