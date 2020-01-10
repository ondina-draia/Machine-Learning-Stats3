import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
  

from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#stock features inside X and classes in Y
df = pd.read_csv('data.csv', low_memory=False) #ficher source des données
X = df.iloc[:,2:]
dl = pd.read_csv('labels.csv')
Y = dl.iloc[:,1]
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
