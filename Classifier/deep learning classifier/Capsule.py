# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:21:24 2021

@author: ice.ice
"""
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import adam_v2 #Adam 改为 adam_v2



def categorical_probas_to_classes(p):
    return np.argmax(p, axis=1)

def to_categorical(y, nb_classes=None):
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1
    return Y

def get_shuffle(data, label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data, label

def calculate_performace(test_num, pred_y,  labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn)/test_num
    precision = float(tp)/(tp + fp + 1e-06)
    npv = float(tn)/(tn + fn + 1e-06)
    sensitivity = float(tp) / (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    mcc = float(tp*tn-fp*fn) / \
        (math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-06)
    f1 = float(tp*2)/(tp*2+fp+fn+1e-06)
    return acc, precision, npv, sensitivity, specificity, mcc, f1

datasetwx = pd.read_csv('train_RPI488_kmer_CT_feature.csv')
datasetwx = np.array(datasetwx)
datasetwx = datasetwx[:, 1:]
datasetwx = np.array(datasetwx,'float32')
[sample_num, input_dimwx] = np.shape(datasetwx)

shapewx = [1,input_dimwx,1]
datasetwx = np.reshape(datasetwx, (sample_num, 1,input_dimwx,1))

label_P = np.ones(int(243))
label_N = np.zeros(int(245))

label = np.hstack((label_P, label_N))
label = np.array(label)


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def Capsule_Layer():
    img = Input(shape=(1,input_dimwx,1))
    x = Conv2D(filters=64, kernel_size=(1,9), strides=2, padding='valid', name='conv1')(img)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    
    x = Conv2D(filters=32, kernel_size=(1,9), strides=2, padding='valid', name='conv1')(img)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    
    """
    NOTE: Capsule architecture starts from here.
    """
    ##### primarycaps coming first ##### 
    x = Conv2D(filters=32, kernel_size=(1,3), strides=2, padding='valid', name='primarycap_conv2')(x)    
    [aa,bb,cc,dd] = x.shape
    numx = int(cc)
    x = Reshape(target_shape=[-1, numx], name='primarycap_reshape')(x)
    x = Lambda(squash, name='primarycap_squash')(x)
    x = BatchNormalization(momentum=0.8)(x)
    
    ##### digitcaps are here ##### 
    x = Flatten()(x)
    uhat = Dense(32, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(32)(c) # compute s_j
    x = Multiply()([uhat, c])
    """
    NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
    """
    s_j = LeakyReLU()(x)
    ##### we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
    c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(32)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(32)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    
    c = Activation('softmax', name='softmax_digitcaps4')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(32)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    pred = Dense(2, activation='sigmoid')(s_j)
    model = Model (img, pred)
    # model.summary()
    model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam (0.0002, 0.5), metrics=['accuracy'])
    return model


sepscores = []
ytest=np.ones((1,2))*0.5
yscore=np.ones((1,2))*0.5

skf= StratifiedKFold(n_splits=5)

for train, test in skf.split(datasetwx, label):
        label_train=to_categorical(label[train])#generate the resonable results
        cv_clf = Capsule_Layer()
        hist=cv_clf.fit(datasetwx[train],label_train,epochs=30)
        
        X = datasetwx
        y = label
        y_test=to_categorical(y[test])#generate the test
        ytest=np.vstack((ytest,y_test))
        y_test_tmp=y[test]
        y_score=cv_clf.predict(X[test])#the output of  probability
        yscore=np.vstack((yscore,y_score))
        fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
        roc_auc = auc(fpr, tpr)
        y_class= categorical_probas_to_classes(y_score)
        acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
        sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
        print('indexes:acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
              % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
        hist=[]
        cv_clf=[]
        
scores=np.array(sepscores)
print("acc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[0]*100,np.std(scores, axis=0)[0]*100))
print("precision=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[1]*100,np.std(scores, axis=0)[1]*100))
print("npv=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[2]*100,np.std(scores, axis=0)[2]*100))
print("sensitivity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[3]*100,np.std(scores, axis=0)[3]*100))
print("specificity=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[4]*100,np.std(scores, axis=0)[4]*100))
print("mcc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[5]*100,np.std(scores, axis=0)[5]*100))
print("f1=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[6]*100,np.std(scores, axis=0)[6]*100))
print("roc_auc=%.2f%% (+/- %.2f%%)" % (np.mean(scores, axis=0)[7]*100,np.std(scores, axis=0)[7]*100))


result1=np.mean(scores,axis=0)
H1=result1.tolist()
sepscores.append(H1)

result=sepscores
data_csv_zhibiao = pd.DataFrame(data=result)
row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)

ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)

data_csv_zhibiao.to_csv('zhibiao_Capsule.csv')
yscore_sum.to_csv('yscore_Capsule.csv')
ytest_sum.to_csv('ytest_Capsule.csv')


