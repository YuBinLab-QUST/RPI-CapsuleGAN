# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:40:51 2021

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
from keras.optimizers import adam_v2

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
datasetwx = np.reshape(datasetwx, (sample_num,input_dimwx))

label_P = np.ones(int(243))
label_N = np.zeros(int(245))

label = np.hstack((label_P, label_N))
label = np.array(label)

def build_discriminator():
    img = Input(shape=(1,input_dimwx,1))
    x = Conv2D(filters=64, kernel_size=(1,9), strides=2, padding='valid', name='conv1')(img)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    
    x = Conv2D(filters=32, kernel_size=(1,9), strides=2, padding='valid', name='conv1')(img)
    x = LeakyReLU()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Flatten()(x)
    pred = Dense(2, activation='sigmoid')(x)
    return Model(img, pred)


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

def build_generator():
    noise_shape =(input_dimwx,)
    x_noise = Input(shape=noise_shape)
    x = Dense(64 * 1 * input_dimwx, activation="relu")(x_noise)
    x = Reshape((1, input_dimwx, 64))(x)
    x = BatchNormalization(momentum=0.2)(x)
    x = UpSampling2D()(x)
    [aa1,bb1,cc1,dd1] = x.shape
    numx1 = int(cc1//4)
    x = Conv2D(32, kernel_size=(2,numx1), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.2)(x)
    [aa2,bb2,cc2,dd2] = x.shape
    numx2 = int(1+cc2-input_dimwx)
    x = Conv2D(16, kernel_size=(1,numx2), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.2)(x)
    x = Conv2D(1, kernel_size=3, padding="same")(x)
    gen_out = Activation("tanh")(x)
    return Model(x_noise, gen_out)

generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(0.002, 0.8), metrics=['accuracy'])

z = Input(shape=(input_dimwx,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(0.002, 0.8), metrics=['accuracy'])

ytest = np.ones((1, 2))*0.5
yscore = np.ones((1, 2))*0.5
sepscores = []
sepscores_ = []

skf = StratifiedKFold(n_splits=5)
for train, test in skf.split(datasetwx, label):
        label_train=to_categorical(label[train])
        X = datasetwx
        y = label
        cv_clf = combined
        hist=cv_clf.fit(X[train],label_train,epochs=30)
        y_test=to_categorical(y[test])
        ytest=np.vstack((ytest,y_test))
        y_test_tmp=y[test]
        y_score=cv_clf.predict(X[test])
        yscore=np.vstack((yscore,y_score))
        fpr, tpr, _ = roc_curve(y_test[:,0], y_score[:,0])
        roc_auc = auc(fpr, tpr)
        y_class= categorical_probas_to_classes(y_score)
        acc, precision,npv, sensitivity, specificity, mcc,f1 = calculate_performace(len(y_class), y_class, y_test_tmp)
        sepscores.append([acc, precision,npv, sensitivity, specificity, mcc,f1,roc_auc])
        print('indexes: acc=%f,precision=%f,npv=%f,sensitivity=%f,specificity=%f,mcc=%f,f1=%f,roc_auc=%f'
              % (acc, precision,npv, sensitivity, specificity, mcc,f1, roc_auc))
        
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

data_csv_zhibiao.to_csv('zhibiao_DCGAN.csv')
yscore_sum.to_csv('yscore_DCGAN.csv')
ytest_sum.to_csv('ytest_DCGAN.csv')



