# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:26:11 2020

@author: Administrator
"""
import scipy.io as scio
import numpy as np
import pandas as pd

 
dataFile = 'RPI1446_P_RPT.mat'
data = scio.loadmat(dataFile)

print(data['RPI1446_P_RPT'])

Data=np.array(data['RPI1446_P_RPT'])
Data_=pd.DataFrame(data=Data)

Data_.to_csv('RPT_RPI1446_P.csv')

#dataFile = 'RPI1807_N_RPT.mat'
#data = scio.loadmat(dataFile)
#
#print(data['RPI1807_N_RPT'])
#
#Data=np.array(data['RPI1807_N_RPT'])
#Data_=pd.DataFrame(data=Data)
#
#Data_.to_csv('RPT_RPI1807_N.csv')

#dataFile = 'RPI488_N_RPT.mat'
#data = scio.loadmat(dataFile)
#
#print(data['RPI488_N_RPT'])
#
#Data=np.array(data['RPI488_N_RPT'])
#Data_=pd.DataFrame(data=Data)
#
#Data_.to_csv('RPT_RPI488_N.csv')