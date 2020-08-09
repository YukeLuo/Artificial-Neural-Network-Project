"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import datetime
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# set random seed
np.random.seed(2048) 
# read dataset and split it into X and Y
dataframe= pd.read_csv("abalone.csv",delimiter=",",header=0)
abalone=dataframe.values
# column 0 is sex in string, it has been changed into integer on column 1 so we don't need column 0 here
X=abalone[:,1:9]
# y variable is ring(age)
Y=abalone[:,9]
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# to scale x and y
X_MinMax = preprocessing.MinMaxScaler()
Y_MinMax = preprocessing.MinMaxScaler()
# reshape x and y. x data has 8 attributes
X = np.array(X).reshape((len(X), 8))
Y = np.array(Y).reshape((len(Y), 1))
x_data = X_MinMax.fit_transform(X)
y_data = Y_MinMax.fit_transform(Y)
# split data into train and test data by 80 and 20.
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=20) 


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
# record the time of start
start = datetime.datetime.now()
# create the model for ANN. Using activation as relu and linear. optimizer as xxx and mrtrics as mae
def create_model():
    model = Sequential()
    # number of nodes for the input layer is 15, dimension of input data is 8, activation is relu
    model.add(Dense(15, input_dim = 8, activation = 'relu'))
    # hidden layer has 10 nodes and relu as activation
    model.add(Dense(10, activation = 'relu'))
    #model.add(Dense(15, activation = 'relu'))
    #model.add(Dense(15, activation = 'relu'))
    # output layer has 1 node and activation as linear
    model.add(Dense(1, activation = 'linear')) 
    # optimizer is adam and metrics is mae
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics = ['mae'])
    #print(model.metrics_names)
    return model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_epochs = 500
batchSize= 32
# create list for mae and mse
mae = []
mse = []
print('model starts') 
# call function that is created before
model = create_model()
# train model with training data, and then test it with test data
history = model.fit(train_x,train_y, epochs=num_epochs, batch_size=batchSize, verbose=1,validation_data=(test_x, test_y))
# show the plot of mse vs epoch
plt.figure()
# plot all data with mse values on y axis and epoch on x axis to see mse changing with epoch increases
plt.plot(history.history['loss'], label='Training error')
plt.plot(history.history['val_loss'], label='Test error')
plt.title('mse values')
plt.ylabel('mse value')
plt.xlabel('Epoch')
plt.legend(loc="upper right")
plt.show()
# access mse and mae and append the data into the lists we created before 
test_mse, test_mae,  = model.evaluate(test_x,test_y)
mae.append(test_mae) 
mse.append(test_mse)
print('test mse:', test_mse)
print('test mae:',  test_mae)
# np.mean is used to calculate the mean value of the list
mean_mae = np.mean(mae)
print('average mse for the model:', np.mean(mse)) 
print('average mae for the model:', mean_mae)
print('minimum mae for the model:', min(mae)) 
# record the end time
end = datetime.datetime.now()
# end time minus start time will be the time duration of the whole training process
completion = end - start
print("Runing Time:",completion) 