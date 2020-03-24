# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:26:02 2020

@author: Yadav
"""
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from tensorflow.nn import sparse_softmax_cross_entropy_with_logits
le = preprocessing.LabelEncoder()
ohe=preprocessing.OneHotEncoder()
def sparse_loss(y_true, y_pred):
    return sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
location="C:\\Users\\Yadav\\Desktop\\Final-project\\drugsTrain.tsv"
loc1="C:\\Users\\Yadav\\Desktop\\Final-project\\drugsTrain.csv"
loc2="C:\\Users\\Yadav\\Desktop\\Final-project\\drugsTest.csv"
loc="C:\\Users\\Yadav\\Desktop\\Final-project\\drugsComTest_raw.tsv"
train =pd.read_csv(location,sep="\t")
test=pd.read_csv(loc,sep="\t")
print(train.shape)
print(train.isna().sum())
train.dropna(inplace=True)
print(train.shape)
print(test.shape)
print(test.isna().sum())
test.dropna(inplace=True)
print(test.shape)
#scaler = MinMaxScaler(feature_range=(0,0.99))
X=train[['condition']]
Y=train[['rating']]
XT=test[['condition']]
XV=test[['rating']]
#print training and testing data
X =X.apply(le.fit_transform)
#X =to_categorical(X)
X=np.array(X).astype(np.float32)
#X = scaler.fit_transform(X)
#print(X.astype)
#Y =le.fit_transform(Y)
#Y=to_categorical(Y)
print(Y)
#Y=normalize(Y)
#Y = scaler.fit_transform(Y)
Y=np.array(Y)
XT =XT.apply(le.fit_transform)
#XT = scaler.fit_transform(XT)
#XT=to_categorical(XT)
XT=np.array(XT).astype(np.float32)
print("XV IS")
#print(XV)
#XV =le.fit_transform(XV)
#XV=to_categorical(XV)
print(XV)
#XV= scaler.fit_transform(XV)
XV=np.array(XV)
#Xm = le.fit_transform(Y)
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.nn import leaky_relu
#from keras.layers.advanced_activations import LeakyReLU
classifier = Sequential()
#First Hidden Layery_pred
classifier.add(Dense(1,kernel_initializer='random_normal'))
#classifier.add(leaky_relu(features=,alpha=0.1))
classifier.add(Dense(16,kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1,kernel_initializer='random_normal'))
classifier.add(Dropout(0.4))
#Fitting the data to the training dataset
classifier.compile(loss='mean_squared_error',optimizer='adamax',metrics=['accuracy'],verbose=1)
classifier.fit(X,Y,epochs=2,verbose=1)
print(classifier.summary())
scores=classifier.evaluate(X, Y)
print('Accuracy on training data: {}% \n Error on training data: {}'.format(scores[1], 1 - scores[1])) 
#scores2=classifier.evaluate(XT, XV)
#print('Accuracy on test data: {}% \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))
#KNeighborsClassifier(input_matrix, n_neighbors=40, p=2, weights=distance)
y_pred=classifier.predict(XT)
print(y_pred)
print(le.inverse_transform(np.argmax(y_pred,axis=1)))
#z=[np.argmax(y_pred,axis=None,out=None) for y in y_pred]
#print(z)
#print(le.inverse_transform(y_pred))
#by=le.inverse_transform(XV)
#az=le.inverse_transform(y_pred)
#print(by)
#print(az)
#print(metrics.confusion_matrix(az,by))
#print(metrics.classification_report(az,by))
#print(metrics.accuracy_score(az,by))