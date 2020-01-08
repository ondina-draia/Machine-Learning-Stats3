
#~ from keras.datasets import mnist
#~ import numpy as np
#~ import matplotlib.pyplot as plt
#~ from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
#~ from keras import Model
import keras
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
import numpy as np
#~ from sklearn.feature_selection import SelectKBest, chi2, f_classif
#~ from sklearn.metrics import accuracy_score
#~ from sklearn.model_selection import train_test_split
#~ from sklearn.preprocessing import StandardScaler



#stock features inside X and classes in Y
df = pd.read_csv('data.csv', low_memory=False) #ficher source des données
X = df.iloc[:,2:]
dl = pd.read_csv('labels.csv')
Y = dl.iloc[:,1]


#Split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33, random_state=42)
