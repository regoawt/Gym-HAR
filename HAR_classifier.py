#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
import csv
import glob
import pickle


# In[20]:


folder_path = 'Data/Next/*.csv'
files = glob.glob(folder_path)

df = pd.concat([pd.read_csv(file, index_col=None) for file in files], axis=0, ignore_index=True)


# In[21]:


data_inputs = np.asarray(df.iloc[:,0:6])

label_encoder = preprocessing.LabelEncoder()
data_labels = np.expand_dims(np.array(label_encoder.fit_transform(data.iloc[:,6])),axis=1)

fileob = open('pickled_label_encoder.obj','wb')
pickle.dump(label_encoder,fileob)
fileob.close()

data_array = np.concatenate((data_inputs,data_labels), axis=1)


# In[32]:


def window_dataset(data,window_size,shift):
    
    num_windows = int(np.floor((len(data)-window_size)/shift))
    windowed_data = np.zeros([num_windows,window_size,np.shape(data)[1]])

    for i in range(num_windows):
        windowed_data[i,:,:] = data[i*shift:i*shift+window_size,:]
    
    return windowed_data


# In[33]:


window_size = 40
shift = 20
windowed_data = window_dataset(data_array,window_size,shift)

shuffled_data = np.take(windowed_data,np.random.rand(windowed_data.shape[0]).argsort(),axis=0)

window_labels = shuffled_data[:,:,6]
mask = np.mean(window_labels, axis=1) == np.min(window_labels, axis=1)

labels = np.mean(window_labels, axis=1)[mask].astype(np.int)

input_data_1 = np.array(shuffled_data[mask,:,0])
input_data_2 = np.array(shuffled_data[mask,:,1])
input_data_3 = np.array(shuffled_data[mask,:,2])
input_data_4 = np.array(shuffled_data[mask,:,3])
input_data_5 = np.array(shuffled_data[mask,:,4])
input_data_6 = np.array(shuffled_data[mask,:,5])
print(input_data_1.shape)


# In[34]:


input_1 = tf.keras.layers.Input(shape=[window_size,1])
conv_1 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_1)
input_2 = tf.keras.layers.Input(shape=[window_size,1])
conv_2 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_2)
input_3 = tf.keras.layers.Input(shape=[window_size,1])
conv_3 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_3)
input_4 = tf.keras.layers.Input(shape=[window_size,1])
conv_4 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_4)
input_5 = tf.keras.layers.Input(shape=[window_size,1])
conv_5 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_5)
input_6 = tf.keras.layers.Input(shape=[window_size,1])
conv_6 = tf.keras.layers.Conv1D(5, 5, activation='relu')(input_6)

concat = tf.keras.layers.Concatenate()([conv_1, conv_2, conv_3, conv_4, conv_5, conv_6])
flatten = tf.keras.layers.Flatten()(concat)
dense1 = tf.keras.layers.Dense(30, activation='relu')(flatten)
dense2 = tf.keras.layers.Dense(30, activation='relu')(dense1)
output = tf.keras.layers.Dense(np.max(labels)+1, activation = 'softmax')(dense2)
model = tf.keras.Model(inputs=[input_1, input_2, input_3, input_4, input_5, input_6], outputs=[output])


# In[35]:


model.summary()


# In[36]:


dummy_labels = tf.keras.utils.to_categorical(labels)


# In[37]:


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit([input_data_1,input_data_2,input_data_3,
                     input_data_4,input_data_5,input_data_6],dummy_labels, epochs = 50,validation_split=0.15)


# In[38]:


model.save('HAR_model')


# In[ ]:




