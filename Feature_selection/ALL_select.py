# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:00:00 2021

@author: ice.ice
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel

def normalize_save(Data):
    """对数据进行标准化化操作"""
    train_data_np = np.array(Data, dtype=float)

    mean = np.mean(train_data_np, axis=0, keepdims=True)
    std = np.std(train_data_np, axis=0, ddof=1, keepdims=True)
    index = np.where(std == 0)  # 防止除数为零
    std[index] = 1e-7
    train_data_np = (train_data_np - mean) / std
    return Data

DATA_SET = 'RPI488'

script_dir, script_name = os.path.split(os.path.abspath(sys.argv[0]))
parent_dir = os.path.dirname(script_dir)
raw_dir = 'E:/Second_Paper/Feature_extraction/Fusion/' 
inputfile = raw_dir + DATA_SET + "_fusion_1367.csv"

############# Data preparition
data_start = pd.read_csv(inputfile)

label_P = np.ones(int('243'))
label_N = np.zeros(int('245'))

label_start = np.hstack((label_P, label_N))
label = np.array(label_start)
data1 = np.array(data_start)
data = data1[:, 1:]
data_nor = normalize_save(data)
# data_nor = scale(data)
# data_nor = data

Zongshu = data_nor
RNA_shu = Zongshu[:,0:674]
pro_shu = Zongshu[:,674:]


# from sklearn.preprocessing import MinMaxScaler
# RNA_shu= pd.DataFrame(data=RNA_shu)
# RNA_shu = MinMaxScaler().fit_transform(RNA_shu.values.reshape(488,674))
# pro_shu= pd.DataFrame(data=pro_shu)
# pro_shu = MinMaxScaler().fit_transform(pro_shu.values.reshape(488,693))


'''卡方进行特征选择'''
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

# def chi2_select(data,label,k=300):
#     model_chi2= SelectKBest(chi2, k=k)
#     new_data=model_chi2.fit_transform(data,label)
#     return new_data

# new_RNA_data = chi2_select(RNA_shu, label, k=39)
# new_pro_data = chi2_select(pro_shu, label, k=135)



'''基于L1正则化的决策树进行特征选择'''
# from sklearn.ensemble import ExtraTreesClassifier

# def selectFromExtraTrees(data,label):
#     clf = ExtraTreesClassifier(n_estimators=500, criterion='gini', max_depth=None, 
#                                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#                                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
#                                 min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=1, 
#                                 random_state=None, verbose=0, warm_start=False, class_weight=None)#entropy
#     clf.fit(data,label)
#     importance=clf.feature_importances_ 
#     model=SelectFromModel(clf,prefit=True)
#     new_data = model.transform(data)
#     return new_data,importance

# new_RNA_data_,index_RNA = selectFromExtraTrees(RNA_shu,label)
# new_pro_data_,index_pro = selectFromExtraTrees(pro_shu,label)

# feature_numbe_RNA = -index_RNA
# H_RNA = np.argsort(feature_numbe_RNA)
# mask_RNA = H_RNA[:39]
# new_RNA_data = RNA_shu[:,mask_RNA]

# feature_numbe_pro = -index_pro
# H_pro = np.argsort(feature_numbe_pro)
# mask_pro = H_pro[:135]
# new_pro_data = pro_shu[:,mask_pro]


'''基于LASSO进行特征选择'''
# from sklearn.linear_model import Lasso,LassoCV

# def lassodimension(data,label,alpha=np.array([0.001])):
#     lassocv=LassoCV(cv=5, alphas=alpha,max_iter=500).fit(data, label)
#     x_lasso = lassocv.fit(data,label)
#     mask = x_lasso.coef_ != 0 
#     new_data = data[:,mask]
#     return new_data,mask 

# def lassodimension2(data,label,alpha=np.array([0.0062])):
#     lassocv=LassoCV(cv=5, alphas=alpha,max_iter=500).fit(data, label)
#     x_lasso = lassocv.fit(data,label)
#     mask = x_lasso.coef_ != 0 
#     new_data = data[:,mask]
#     return new_data,mask 

# new_RNA_data,index_RNA=lassodimension(RNA_shu,label)
# new_pro_data,index_pro=lassodimension2(pro_shu,label)


'''基于LinearSVC进行特征选择'''
# from sklearn.svm import LinearSVC,SVC

# def LinearSVC_select(data,label,lamda):
#     lsvc = LinearSVC(C=lamda, penalty="l1", dual=False).fit(data,label)
#     model = SelectFromModel(lsvc,prefit=True)
#     new_data= model.transform(data)
#     return new_data

# new_RNA_data = LinearSVC_select(RNA_shu,label,lamda=5)
# new_pro_data = LinearSVC_select(pro_shu,label,lamda=0.45)


'''基于LLE进行特征选择'''
# from sklearn.manifold import LocallyLinearEmbedding

# def LLE(data,n_components=300):
#     embedding = LocallyLinearEmbedding(n_components=n_components)
#     X_transformed = embedding.fit_transform(data)
#     return X_transformed

# new_RNA_data = LLE(RNA_shu,n_components=39)
# new_pro_data = LLE(pro_shu,n_components=135)


'''基于Lasso中的正交匹配追踪OMP进行特征选择'''
# from sklearn.linear_model import OrthogonalMatchingPursuit
# def omp_omp(data,label,n_nonzero_coefs=100):
#     omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
#     omp.fit(data, label)
#     coef = omp.coef_
#     idx_r, = coef.nonzero()
#     new_data=data[:,idx_r]
#     return new_data,idx_r

# new_RNA_data,index_RNA = omp_omp(RNA_shu,label,n_nonzero_coefs=39)
# new_pro_data,index_pro = omp_omp(pro_shu,label,n_nonzero_coefs=135)


'''基于TSVD进行特征选择'''
# from sklearn.decomposition import TruncatedSVD

# def TSVD(data,n_components=300):
#     svd = TruncatedSVD(n_components=n_components)
#     new_data=svd.fit_transform(data)  
#     return new_data

# new_RNA_data = TSVD(RNA_shu,n_components=39)
# new_pro_data = TSVD(pro_shu,n_components=135)


'''基于SE进行特征选择'''
# from sklearn.manifold import SpectralEmbedding 

# def SE_select(data,n_components=300):
#     embedding = SpectralEmbedding(n_components=n_components)
#     X_transformed = embedding.fit_transform(data)
#     return X_transformed

# new_RNA_data = SE_select(RNA_shu, n_components=39)
# new_pro_data = SE_select(pro_shu, n_components=135)


'''基于MI进行特征选择'''
# from sklearn.feature_selection import mutual_info_classif

# def mutual_mutual(data,label,k=300):
#     model_mutual= SelectKBest(mutual_info_classif, k=k)
#     new_data=model_mutual.fit_transform(data, label)
#     mask = model_mutual._get_support_mask()
#     return new_data,mask

# new_RNA_data ,index_RNA = mutual_mutual(RNA_shu,label,k=39)
# new_pro_data ,index_pro = mutual_mutual(pro_shu,label,k=135)


'''基于MDS进行特征选择'''
from sklearn.manifold import MDS

def MDS_select(data,n_components=300):
    embedding = MDS(n_components=n_components)
    new_data = embedding.fit_transform(data)
    return new_data

new_RNA_data = MDS_select(RNA_shu,n_components=39)
new_pro_data = MDS_select(pro_shu,n_components=135)




'''savepath'''
[m1,n1] = np.shape(new_RNA_data)
[m2,n2] = np.shape(new_pro_data)
cc1 = str(n1)
cc2 = str(n2)
data_new = np.hstack((new_RNA_data,new_pro_data))
optimal_features = pd.DataFrame(data=data_new)
savep = '_MDS_'+cc1+'_'+cc2 +'.csv'
outpath = 'E:\Second_Paper\Compare\Selection/'
optimal_features.to_csv(outpath + DATA_SET + savep)
