"""
Most of this code is adapted from "Unsupervised Motion Artifact Detection in Wrist-Measured EDA"
Creates a feature matrix from a dataset of EDA and accelerometer to be used in machine learning
"""

import numpy as np
import pandas as pd
from statistics import mode
from scipy import stats
import pywt


dataFiles = 'John.csv'

dataFiles_labels = 'John_Epoch.csv'
nSubjects_John = 7; # Number of time segments in John's dataset

def statistics(data): # The function for the 4 statistics
    avg = np.mean(data) # mean
    sd = np.std(data) # standard deviation
    maxm = max(data) # maximum
    minm = min(data) # minimum
    return avg,sd,maxm,minm
    
def Derivatives(data): # Get the first and second derivatives of the data
    deriv = (data[1:-1] + data[2:])/ 2. - (data[1:-1] + data[:-2])/ 2.
    secondDeriv = data[2:] - 2*data[1:-1] + data[:-2]
    return deriv,secondDeriv

def featureMatrix(data,labels_all): # Construct the feature matrix
    length = len(labels_all)
    # Create the one label set by the majority vite
    labels = np.zeros((length,))
    labels=(labels_all)
    # Divide the data into time windows, 40 data points each time window.
    EDA = data[0:length*40,9].reshape(length,40)
    ACCx = data[0:length*40,1].reshape(length,40)
    ACCy = data[0:length*40,2].reshape(length,40)
    ACCz = data[0:length*40,3].reshape(length,40)
    # Get the ACC magnitude by root-mean-square
    acc = np.sqrt(np.square(data[0:length*40,1]) + np.square(data[0:length*40,2]) + np.square(data[0:length*40,3]))
    ACC = acc.reshape(length,40)
    # Construct the feature matrix, 24 EDA features, 96 ACC features, and 120 features in total. 
    features = np.zeros((length,120))
    for i in range(length):
        deriv_EDA,secondDeriv_EDA = Derivatives(EDA[i,:])
        deriv_ACC,secondDeriv_ACC = Derivatives(ACC[i,:])
        deriv_ACCx,secondDeriv_ACCx = Derivatives(ACCx[i,:])
        deriv_ACCy,secondDeriv_ACCy = Derivatives(ACCy[i,:])
        deriv_ACCz,secondDeriv_ACCz = Derivatives(ACCz[i,:])
        _, EDA_cD_3, EDA_cD_2, EDA_cD_1 = pywt.wavedec(EDA[i,:], 'Haar', level=3) #3 = 1Hz, 2 = 2Hz, 1=4Hz
        _, ACC_cD_3, ACC_cD_2, ACC_cD_1 = pywt.wavedec(ACC[i,:], 'Haar', level=3) 
        _, ACCx_cD_3, ACCx_cD_2, ACCx_cD_1 = pywt.wavedec(ACCx[i,:], 'Haar', level=3) 
        _, ACCy_cD_3, ACCy_cD_2, ACCy_cD_1 = pywt.wavedec(ACCy[i,:], 'Haar', level=3) 
        _, ACCz_cD_3, ACCz_cD_2, ACCz_cD_1 = pywt.wavedec(ACCz[i,:], 'Haar', level=3) 
        
        ### EDA features
        # EDA statistical features:
        features[i,0:4] = statistics(EDA[i,:])
        features[i,4:8] = statistics(deriv_EDA)
        features[i,8:12] = statistics(secondDeriv_EDA)
        # EDA wavelet features:
        features[i,12:16] = statistics(EDA_cD_3)
        features[i,16:20] = statistics(EDA_cD_2)
        features[i,20:24] = statistics(EDA_cD_1)
        
        ### ACC features
        ## ACC statistical features:
        # Acceleration magnitude:
        features[i,24:28] = statistics(ACC[i,:])
        features[i,28:32] = statistics(deriv_ACC)
        features[i,32:36] = statistics(secondDeriv_ACC)
        # Acceleration x-axis:
        features[i,36:40] = statistics(ACCx[i,:])
        features[i,40:44] = statistics(deriv_ACCx)
        features[i,44:48] = statistics(secondDeriv_ACCx)
        # Acceleration y-axis:
        features[i,48:52] = statistics(ACCy[i,:])
        features[i,52:56] = statistics(deriv_ACCy)
        features[i,56:60] = statistics(secondDeriv_ACCy)
        # Acceleration z-axis:
        features[i,60:64] = statistics(ACCz[i,:])
        features[i,64:68] = statistics(deriv_ACCz)
        features[i,68:72] = statistics(secondDeriv_ACCz)
        ## ACC wavelet features:
        # ACC magnitude wavelet features:
        features[i,72:76] = statistics(ACC_cD_3)
        features[i,76:80] = statistics(ACC_cD_2)
        features[i,80:84] = statistics(ACC_cD_1)
        # ACC x-axis wavelet features:
        features[i,84:88] = statistics(ACCx_cD_3)
        features[i,88:92] = statistics(ACCx_cD_2)
        features[i,92:96] = statistics(ACCx_cD_1)
        # ACC y-axis wavelet features:
        features[i,96:100] = statistics(ACCy_cD_3)
        features[i,100:104] = statistics(ACCy_cD_2)
        features[i,104:108] = statistics(ACCy_cD_1)
        # ACC z-axis wavelet features:
        features[i,108:112] = statistics(ACCz_cD_3)
        features[i,112:116] = statistics(ACCz_cD_2)
        features[i,116:120] = statistics(ACCz_cD_1)
        
    featuresAll = stats.zscore(features) # Normalize the data using z-score
    featuresAcc = featuresAll[:,24:120] # 96 ACC features
    featuresEda = featuresAll[:,0:24] #24 EDA features
    return featuresAll,featuresAcc,featuresEda,labels

# Load the data and construct the feature matrix
data_John   = dict()
labels_John = dict()
JohnAll = dict()
JohnAcc = dict()
JohnEda = dict()
JohnLabels = dict()
JohnGroups = dict()

data_John = np.loadtxt(dataFiles, delimiter=',', skiprows=2)
labels_John = np.loadtxt(dataFiles_labels, delimiter=',', skiprows=1, usecols=(1))
labels_John[labels_John==0]=1 # Assume the unlabeled time windows as clean
labelsJohn = labels_John-1 # Make the labels include only 0s and 1s
JohnGroups = np.ones(len(labels_John)) # The group number for the leave one group out cross-validation
JohnAll,JohnAcc,JohnEda,JohnLabels = featureMatrix(data_John,labels_John)


#saving feature matrices for machine learning
np.savetxt('JohnAll.csv', JohnAll, delimiter=',')
np.savetxt('JohnAcc.csv', JohnAcc, delimiter=',')
np.savetxt('JohnEda.csv', JohnEda, delimiter=',')
np.savetxt('JohnLabels.csv', JohnLabels, delimiter=',')