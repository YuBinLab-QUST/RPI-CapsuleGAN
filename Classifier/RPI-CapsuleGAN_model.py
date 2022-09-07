# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 15:27:44 2021

@author: ice.ice
"""
import os,sys,math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.activations import sigmoid
from keras.layers import Input, Dense, Layer, Reshape, Flatten
from keras.layers import multiply, Add, Permute
from keras.layers import Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

# DATA_SET = 'RPI1807'
# script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
# parent_dir = os.path.dirname(script_dir)
# inputpath = parent_dir +'/Feature_selection/'
# # inputfile =  DATA_SET + "_Adaptive_LASSO_0.0005.csv"
# inputfile =  inputpath + DATA_SET + "_T488_EN_30_62.csv"

# datasetwx = pd.read_csv(inputfile)
datasetwx = pd.read_csv('RPI488_EN_0.09_37_143.csv')
datasetwx = np.array(datasetwx)
datasetwx = datasetwx[:, 1:]
datasetwx = np.array(datasetwx,'float32')
[sample_num, input_dimwx] = np.shape(datasetwx)
# datasetwx = np.reshape(datasetwx, (sample_num, 1,input_dimwx,1))
shapewx = [1,input_dimwx,1]

# label_P = np.ones(int(369))
# label_N = np.zeros(int(369))
# label_P = np.ones(int(1446))
# label_N = np.zeros(int(1560))
# label_P = np.ones(int(1807))
# label_N = np.zeros(int(1436))
label_P = np.ones(int(243))
label_N = np.zeros(int(245))
# label_P = np.ones(int(7317))
# label_N = np.zeros(int(7317))
# label_P = np.ones(int(2241))
# label_N = np.zeros(int(2241))
label = np.hstack((label_P, label_N))
label = np.array(label)


def get_shuffle(data,label):    
    #shuffle data
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    data = data[index]
    label = label[index]
    return data,label 

(datasetwx,label) = get_shuffle(datasetwx,label)



def normalize_save(Data):
    """对数据进行标准化化操作"""
    train_data_np = np.array(Data, dtype=float)

    mean = np.mean(train_data_np, axis=0, keepdims=True)
    std = np.std(train_data_np, axis=0, ddof=1, keepdims=True)
    index = np.where(std == 0)  # 防止除数为零
    std[index] = 1e-7
    train_data_np = (train_data_np - mean) / std
    return Data

#################################################### Data preparition

# DATA_SET = 'RPI369'
# # script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
# # parent_dir = os.path.dirname(script_dir)
# # inputpath = parent_dir +'/Feature_selection/'
# # inputfile =  inputpath + DATA_SET + "_Adaptive_LASSO_0.0008.csv"

# datasetwx = pd.read_csv('RPI369_EN_30_62.csv')
# # datasetwx = pd.read_csv("RPI488_GTPC_1736D_Group_Lasso0.01.csv")
# # datasetwx = pd.read_csv("RPI488_1736DAdaptive_LASSO_0.001.csv")
# # datasetwx = pd.read_csv("RPI1446_1736DAdaptive_LASSO_0.001.csv")
# # datasetwx = pd.read_csv("RPI1807_1736DAdaptive_LASSO_0.001.csv")
# # datasetwx = pd.read_csv("RPI488_lasso0.01.csv")
# # datasetwx = pd.read_csv('RPIxxx.csv')
# datasetwx = np.array(datasetwx)
# datasetwx = datasetwx[:, 1:]

# # datasetwx = normalize_save(datasetwx)

# datasetwx = np.array(datasetwx,'float32')
# [sample_num, input_dimwx] = np.shape(datasetwx)
# # datasetwx = np.reshape(datasetwx, (sample_num, 1,input_dimwx,1))
# shapewx = [1,input_dimwx,1]

# label_P = np.ones(int(369))
# label_N = np.zeros(int(369))
# # label_P = np.ones(int(1446))
# # label_N = np.zeros(int(1560))
# # label_P = np.ones(int(1807))
# # label_N = np.zeros(int(1436))
# # label_P = np.ones(int(243))
# # label_N = np.zeros(int(245))


# label = np.hstack((label_P, label_N))
# label = np.array(label)


def scale_mean_var(input_arr,axis=0):
    #from sklearn import preprocessing
    #input_arr= preprocessing.scale(input_arr.astype('float'))
    mean_ = np.mean(input_arr,axis=0)
    scale_ = np.std(input_arr,axis=0)
    #减均值 
    output_arr= input_arr- mean_
    #判断均值是否接近0
    mean_1 = output_arr.mean(axis=0)
    if not np.allclose(mean_1, 0):
        output_arr -= mean_1
    #将标准差为0元素的置1
    #scale_ = _handle_zeros_in_scale(scale_, copy=False)
    scale_[scale_ == 0.0] = 1.0
    #除以标准差
    output_arr /=scale_
    #再次判断均值是否为0
    mean_2 = output_arr .mean(axis=0)
    if not np.allclose(mean_2, 0):
        output_arr  -= mean_2

    return output_arr

########################################################### Def Cbam
def channel_attention(input_feature, ratio=8):
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	shared_layer_one = Dense(channel//ratio,
							 kernel_initializer='he_normal',
							 activation = 'relu',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('hard_sigmoid')(cbam_feature)
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	return multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
	kernel_size = 7
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature._keras_shape[-1] == 1
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	return multiply([input_feature, cbam_feature])


def cbam_block(cbam_feature,ratio=8):
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature, )
	return cbam_feature


############################################## Def discriminator and generator
def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def build_discriminator():
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
    uhat = Dense(128, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)
    c = Activation('softmax', name='softmax_digitcaps1')(uhat) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(128)(c) # compute s_j
    x = Multiply()([uhat, c])
    """
    NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
    """
    s_j = LeakyReLU()(x)
    ##### we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
    c = Activation('softmax', name='softmax_digitcaps2')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(128)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)

    c = Activation('softmax', name='softmax_digitcaps3')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(128)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    
    c = Activation('softmax', name='softmax_digitcaps4')(s_j) # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
    c = Dense(128)(c) # compute s_j
    x = Multiply()([uhat, c])
    s_j = LeakyReLU()(x)
    # ##### preparition for cbam_block
    s_j = Reshape((-1,128,1))(s_j)
    inputs = s_j
    residual = Conv2D(filters=64, kernel_size=(1,1), strides=1, padding='same', name='convxxx')(inputs)
    residual = BatchNormalization(momentum=0.8)(residual)
    
    cbam = cbam_block(residual)
    # cbam = channel_attention(residual)
    # cbam = spatial_attention(residual)
    
    cbam = Reshape((-1,))(cbam)
    pred = Dense(2, activation='sigmoid')(cbam)
    
    # cbam = Reshape((-1,))(s_j)
    # pred = Dense(2, activation='sigmoid')(cbam)
    
    
    return Model(img, pred)


# build and compile the discriminator
discriminator = build_discriminator()
# print('DISCRIMINATOR:')
# discriminator.summary()
# discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
discriminator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


# generator structure
def build_generator():
    """
    Generator follows the DCGAN architecture and creates generated image representations through learning.
    """
    noise_shape =(input_dimwx,)
    x_noise = Input(shape=noise_shape)
    # we apply different kernel sizes in order to match the original image size
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
    #### x = UpSampling2D()(x)
    numx2 = int(1+cc2-input_dimwx)
    x = Conv2D(16, kernel_size=(1,numx2), padding="valid")(x)
    x = Activation("relu")(x)
    x = BatchNormalization(momentum=0.2)(x)
    x = Conv2D(1, kernel_size=3, padding="same")(x)
    gen_out = Activation("tanh")(x)
        
    return Model(x_noise, gen_out)


# build and compile the generator
generator = build_generator()
# print('GENERATOR:')
# generator.summary()
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.8), metrics=['accuracy'])
# generator.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# feeding noise to generator
z = Input(shape=(input_dimwx,))
img = generator(z)
# for the combined model we will only train the generator
discriminator.trainable = False
# try to discriminate generated images
valid = discriminator(img)
# the combined model (stacked generator and discriminator) takes
# noise as input => generates images => determines validity 
combined = Model(z, valid)
# print('COMBINED:')
# combined.summary()
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.8), metrics=['accuracy'])
# combined.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


####################################################   Prepartion for Train or Test

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
        # hist=cv_clf.fit(X[train],label_train,epochs=3)
        y_test=to_categorical(y[test])
        
        
        history=cv_clf.fit(X[train],label_train,epochs=30,
                           validation_data=(X[test],y_test))
        # epochs = len(history.history['loss'])


        
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

# data_csv_zhibiao = pd.DataFrame(data=result)
# data_csv_zhibiao.to_csv('RPI369_zhibiao.csv')


row=yscore.shape[0]
yscore=yscore[np.array(range(1,row)),:]
yscore_sum = pd.DataFrame(data=yscore)
#### yscore_sum.to_csv('yscore_sum_DNN.csv')
ytest=ytest[np.array(range(1,row)),:]
ytest_sum = pd.DataFrame(data=ytest)
#### ytest_sum.to_csv('ytest_sum_DNN.csv')
fpr, tpr, _ = roc_curve(ytest[:,0], yscore[:,0])
auc_score=np.mean(scores, axis=0)[7]
lw=2
plt.plot(fpr, tpr, color='darkorange',
lw=lw, label='ROC (area = %0.2f%%)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.03, 1.03])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("ROC.jpg")
plt.show()
# data_csv_zhibiao = pd.DataFrame(data=result)
# data_csv_zhibiao.to_csv('RPI1807_zhibiao.csv')


history_dict = history.history
# epochs = len(history.history['loss'])
# plt.plot(range(0,epochs,1), history_dict['loss'], label='train_loss')
# plt.plot(range(0,epochs,1), history_dict['val_loss'], label='test_loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend()
# # plt.savefig("A-Res/Unet-过拟合C0.jpg")
# plt.show()


##############################################
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'darkorange', label='Training loss')
plt.plot(epochs, val_loss_values, 'navy', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("TTloss.jpg")
plt.show()
##################################################




train_acc = history_dict['acc']
val_acc = history_dict['val_acc']

# plt.plot(range(0,epochs,1), train_acc, 'bo', label='Training acc')
# plt.plot(range(0,epochs,1), val_acc, 'b', label='Validation acc')

plt.plot(epochs, train_acc, 'darkorange', label='Training acc')
plt.plot(epochs, val_acc, 'navy', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("TTacc.jpg")
plt.show()




