# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:19:38 2022

@author: HP
"""

#Import necessary packages
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import matplotlib.pyplot as plt
import datetime
import os

#1. Read CSV data
file_path = r"C:\Users\HP\Documents\TensorFlow\datasets\garments_worker_productivity.csv"
garment_data = pd.read_csv(file_path)

#%%
#2. Data cleaning
#(a) Drop less useful column
garment_data['date'] = pd.to_datetime(garment_data['date'])
garment_data['date'] = garment_data['date'].dt.day
#(b) Replace missing values in wip column
garment_data.wip = garment_data.wip.fillna(0)

#(c) Replace missing value
garment_data['department'] = garment_data['department'].replace(['finishing '],['finishing'])
garment_data['department'] = garment_data['department'].replace(['sweing'],['sewing'])

print(garment_data.isna().sum())

#%%

#Team is a categorical feature, so change its type to string
garment_data['team'] = garment_data['team'].astype(str)
#%%
#Target productivity has 2 decimal digits. Productivity may fluctuate. So, we can check if the actual productivity misses target + 0.001
X = garment_data.drop(columns=['actual_productivity', 'idle_time', 'idle_men', 'date'])
y = garment_data['actual_productivity'] < garment_data['targeted_productivity'] + 0.001

y.value_counts()

#%%
#Dataset is imbalanced. So we have to use stratify=y during split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=0)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,stratify=y_train,random_state=0)

#%%
#3. Split the data into features and label
garment_data = garment_data.drop('date',axis=1)
X = garment_data.copy()
y = X.pop('actual_productivity')

#4. Check the data
print("------------------Features-------------------------")
print(X.head())
print("-----------------Label----------------------")
print(y.head())


#%%
#5. Ordinal encode categorical features
quarter_categories = ['Quarter1','Quarter2','Quarter3','Quarter4','Quarter5']
team_categories = ['1','2','3','4','5','6','7','8','9','10','11','12']
day_categories = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
department_categories = ['sewing','finishing']
ordinal_encoder = OrdinalEncoder(categories=[quarter_categories,department_categories,team_categories,day_categories])
X[['quarter','department','team','day']] = ordinal_encoder.fit_transform(X[['quarter','department','team','day']])
#Check the transformed features
print("---------------Transformed Features--------------------")
print(X.head())

#%%
#6. Split the data into train-validation-test sets, with a ratio of 60:20:20
SEED = 12345
X_train,X_iter,y_train,y_iter = train_test_split(X,y,test_size=0.4,random_state=SEED)
X_val,X_test,y_val,y_test = train_test_split(X_iter,y_iter,test_size=0.5,random_state=SEED)
#%%

#7. Perform feature scaling, using training data for fitting
standard_scaler = StandardScaler()
standard_scaler.fit(X_train)
X_train = standard_scaler.transform(X_train)
X_val = standard_scaler.transform(X_val)
X_test = standard_scaler.transform(X_test)

#Data preparation is completed at this step

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = X_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=number_input))
model.add(tf.keras.layers.Dense(128,activation='elu'))
model.add(tf.keras.layers.Dense(64,activation='elu'))
model.add(tf.keras.layers.Dense(32,activation='elu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))

#9.Compile the model
model.compile(optimizer='adam',loss='mse',metrics=['mae','mse'])

#%%
#10. Train and evaluate the model with validation data
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\HP\Documents\Program AI06\p2_log"
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
EPOCHS = 100
BATCH_SIZE= 64
history = model.fit(X_train,y_train,validation_data=(X_val,y_val),batch_size=BATCH_SIZE,epochs=EPOCHS,callbacks=[tb_callback,es_callback])


#%%
#11. Evaluate with test data for wild testing
test_result = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test MAE = {test_result[1]}")
print(f"Test MSE = {test_result[2]}")

#12. Plot a graph of prediction vs label on test data
predictions = np.squeeze(model.predict(X_test))
labels = np.squeeze(y_test)
plt.plot(predictions,labels,".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels with Test Data")
save_path = r"C:\Users\HP\Documents\GitHub\ai06-test-repo-2\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()