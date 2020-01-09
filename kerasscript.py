
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
from sklearn.feature_selection import SelectKBest, chi2, f_classif
#~ from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#~ from sklearn.preprocessing import StandardScaler



#stock features inside X and classes in Y
df = pd.read_csv('data.csv', low_memory=False) #ficher source des données
X = df.iloc[:,2:]
dl = pd.read_csv('labels.csv')
Y = dl.iloc[:,1]

print(Y)

# Feature selection
fts = SelectKBest(f_classif, k=12).fit(X, Y)
# Obtention of a numpy.ndarray: contains bool like positions of the selected columns 
support = fts.get_support()  
# Feature filter 
new_X = fts.transform(X)


#Split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(new_X,Y, test_size = 0.33, random_state=42)

X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))


noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)


n = 5
plt.figure(figsize=(20, 2))
for i in range(n):
	ax = plt.subplot(1, n, i+1)
	plt.gray()
	ax.set_xlabel('Features')
	ax.set_ylabel('Training and Testing accuracy')
plt.show()
