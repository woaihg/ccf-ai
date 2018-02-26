#coding: utf-8
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.layers import GlobalAveragePooling1D,MaxPooling1D,Flatten
from keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
max_features = 600000
maxlen = 20000  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
embedding_dims = 256
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

x_train = np.load("data/train.npy")
x_test = np.load("data/test.npy")
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

label = pd.read_csv("data/train_money_label.csv")
y_train = to_categorical(label.as_matrix(), num_classes=9)
trainrow=pd.read_csv("data/trainrow.csv")
testrow = pd.read_csv("data/testrow.csv")

model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.15))
model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=32,epochs=1)

weights=[]
for i in range(len(model.layers)):
    weights.append(model.layers[i].get_weights())

K.clear_session()

#得到输出特征
model = Sequential()
model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
model.add(Dropout(0.15))
model.add(Conv1D(250,3,padding='valid',activation='relu',strides=1))
model.add(GlobalMaxPooling1D())
for i in range(len(model.layers)):
    model.layers[i].set_weights(weights[i])

trainmf = model.predict(x_train)
testmf = model.predict(x_test)



#附加特征
trainappd=[]
file = open("data/trainjine.txt",encoding='utf-8')
for line in file:
	trainappd.append([float(line[0:-1])])

i=0
file = open("data/trainjinemax.txt",encoding='utf-8')
for line in file:
	trainappd[i].append(float(line[0:-1]))
	i=i+1

trainappd = np.array(trainappd)
trainmf=np.concatenate([trainmf,trainappd],axis=1)

np.save("data/mtrain1",trainmf)

testappd=[]
file = open("data/testjine.txt",encoding='utf-8')
for line in file:
    testappd.append([float(line[0:-1])])

i=0
file = open("data/testjinemax.txt",encoding='utf-8')
for line in file:
    testappd[i].append(float(line[0:-1]))
    i=i+1

testappd = np.array(testappd)
testmf=np.concatenate([testmf,testappd],axis=1)
np.save("data/mtest1",testmf)


a=np.load('data/trainlaw.npy')
xtrain=np.concatenate([trainmf,a],axis=1)
b=np.load('data/testlaw.npy')
xtest=np.concatenate([testmf,b],axis=1)
#xgb训练
import xgboost as xgb
params = {
            'objective': 'multi:softmax',
            'eta': 0.05,
            'max_depth': 12,
            'eval_metric': 'merror',
            'num_class':9,
            'missing': 0,
            'silent' : 1,
            'nthread':10
            }

c=label.as_matrix()
xgbtrain = xgb.DMatrix(xtrain,c)
xgbtest = xgb.DMatrix(xtest)
watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
num_rounds=120
model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=120)

ans=[]
probas=model.predict(xgbtest)
for i in range(len(probas)):
    ans.append([testrow.iloc[i],probas[i]])

ansD = pd.DataFrame(ans,columns=['row_id','money'],index=None)
ansD.to_csv("data/money-12-2.csv",index=None)











