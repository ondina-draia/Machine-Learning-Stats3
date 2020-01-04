#dataset: collection données RNA-seq, extraction random d'expressions
#de gènes de patients ayant des tumeurs differentes: BRCA; KIRC, COAD, LUAD, PRAD

# les samples/ instances stockés row-wise. 

#dummy-name, gene_XX donné à chaque attribut

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import linear_model

df = pd.read_csv('data.csv', low_memory=False) #ficher source des données
dl = pd.read_csv('labels.csv')
#labels = dl['Class'].tolist()
df['Class'] = dl['Class']
df.set_index('Class','Unnamed: 0')
#del df['Unnamed: 0']
#print(df.head())
#stocker les features dans X et les classes dans Y
Y = df.iloc[:,-1]
X = df.loc[:, df.columns != 'Class']
del X['Unnamed: 0']

#Split training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

#Create a simple model
logreg = linear_model.LogisticRegression(max_iter=1500, random_state = 0)

#Train the model
logreg.fit(X_train_std,Y_train)
Z = logreg.predict(X)

print(pd.crosstab(Y,Z))

# Decision region drawing
from matplotlib.colors import ListedColormap

#print(Y) Name: Class, Length: 801, dtype: object
#print(X) [801 rows x 20531 columns]

print(df)

#En construction
#sns.set_style("whitegrid")
#sns.FacetGrid(df, hue="Class", height=4) \
#    .map(plt.scatter, "Unnamed: 0", "Class") \
#    .add_legend()
#plt.show()