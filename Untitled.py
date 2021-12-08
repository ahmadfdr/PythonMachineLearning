#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6


# In[2]:


df = pd.read_excel('TKTK1.xlsx', sheet_name='KOPI SUSU')
df


# In[3]:


df.shape


# In[4]:


df1 = df.reset_index()['Penjualan Kopi Susu']
df1


# In[5]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[6]:


#LSTM is sensitive to the scale of the data, so we apply MinMax Scaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# In[7]:


scaler = MinMaxScaler(feature_range = (0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
df1


# In[8]:


#Splitting the data into train and test split
training_size = int(len(df1)*0.75)
test_size = len(df)-training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]
training_size, test_size


# In[9]:


train_data


# In[59]:


import numpy
#Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range (len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   #i=0, 0,1,2,3----9   10
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[124]:


#Reshape into X=t, t+1, t+2, t+3, and Y=t+4
time_step = 1
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


# In[125]:


print(X_train.shape), print(y_train.shape)


# In[126]:


print(X_test.shape), print(ytest.shape)


# In[127]:


#Reshape input to be [samples, time_steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[128]:


#Create the stacked LSTM Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[147]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')


# In[148]:


model.summary()


# In[156]:


model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=10, batch_size=1, verbose=1)


# In[157]:


import tensorflow as tf


# In[158]:


#Do the prediction and check performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[159]:


#Transformback to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[160]:


#Calculate RSME perfomance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train, train_predict))


# In[161]:


#Test data RSME
math.sqrt(mean_squared_error(ytest, test_predict))


# In[162]:


###Plotting
#Shift train prediction for plotting
look_back = 1
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(train_predict)+look_back, :] = train_predict
#Shift test prediction for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
#Plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot, label='Train Data')
plt.plot(testPredictPlot, label='Test Data')
plt.legend(loc='best')
plt.show()


# In[163]:


len(test_data)


# In[164]:


x_input = test_data[23].reshape(1,-1)
x_input.shape


# In[165]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()
temp_input


# In[166]:


#Demonstrate prediction next 10 days
from numpy import array
lst_output = []
n_steps = 1
i = 0
while(i<30):
    if (len(temp_input)>1):
        #print temp_input
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1, n_steps, 1))
        #print x_input
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print temp_input
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

print(lst_output)


# In[168]:


day_new = np.arange(1,2)
day_pred = np.arange(2,32)


# In[169]:


import matplotlib.pyplot as plt


# In[170]:


len(df1)


# In[171]:


# df3=df1.tolist()
# df3.extend(lst_output)


# In[172]:


plt.plot(day_new, scaler.inverse_transform(df1[92:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))


# In[ ]:


# df3=df1.tolist()
# df3.extend(lst_output)
# plt.plot(df3[100:])

