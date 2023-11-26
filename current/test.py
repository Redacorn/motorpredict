import sys
sys.path.append(r"C:\Users\chara\OneDrive\문서\GitHub\motorpredict\ref\current")

import warnings
warnings.filterwarnings('ignore')

import time
import os
import glob
from time import sleep
import datetime
import pytz
import argparse

import numpy as np
import pandas as pd
import pickle
import joblib

import scipy
import sklearn
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from Current_Feature_Extractor import Extract_Time_Features, Extract_Phase_Features, Extract_Freq_Features


filename = r"C:\Users\chara\OneDrive\문서\GitHub\motorpredict\ref\STFCB-20201012-0105-0138_20201113_074205_002.csv"
datapath = r"D:\기계시설물 고장 예지 센서\Training\current\2.2kW\L-DEF-01\정상"
savepath = r"D:\기계시설물 고장 예지 센서\Training\current\2.2kW\L-DEF-01\transform"


def data_transform(filename):
    try:
        cur = pd.read_csv(filename, header=None, skiprows=9)
        cur = np.asarray(cur)[:,1:4].transpose()

        meta = pd.read_csv(filename, header=None, skiprows=4, nrows=1)
        rpm = int(meta[2])

        fs = pd.read_csv(filename, header=None, skiprows=6, nrows=1)
        Fs = int(fs[1])

        TimeFeatureExtractor = Extract_Time_Features(cur)
        features_time = TimeFeatureExtractor.Features().flatten()

        PhaseFeatureExtractor = Extract_Phase_Features(cur, Fs)
        features_phase = PhaseFeatureExtractor.Features().flatten()

        FreqFeatureExtractor = Extract_Freq_Features(cur, rpm, Fs)
        features_freq = FreqFeatureExtractor.Features().flatten()

        features = np.concatenate((features_time, features_phase, features_freq))
        data = pd.DataFrame(features).T

        data.columns = ['R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                'R_Impulse', 'S_Impulse', 'T_Impulse',
                'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']

        data['WATT'] = meta[3]
        y = pd.read_csv(filename, header=None, skiprows=3, nrows=1)
        y = y[1]

    except (NameError, IndexError, TypeError, pd.errors.EmptyDataError):
        print('Source Data Error')
        
    else:
        df = data[['WATT', 'R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                    'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                    'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                    'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                    'R_Impulse', 'S_Impulse', 'T_Impulse',
                    'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                    'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                    'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']]

        #norm = np.load('current_norm.npy')#norm
        #X = (df - norm[0]) / norm[1]
    # save df
    df.to_csv(savepath + '/' + os.path.basename(filename), index=False)

    
for i, file in enumerate(glob.glob(datapath + '/*.csv')):
    # print progress with ETA
    print("Progress: {}/{} {:.2f}%".format(i+1, len(glob.glob(datapath + '/*.csv')), (i+1)/len(glob.glob(datapath + '/*.csv'))*100), end='\r')

    data_transform(file)