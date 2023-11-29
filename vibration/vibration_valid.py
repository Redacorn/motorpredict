import random
import sys
import time
import os
import glob
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
from Vibration_Feature_Extractor import Extract_Time_Features, Extract_Freq_Features

def framework():
    print('########## 진동 유효 검증 시작 #############')
    print('시작시간(UCT+09:00)',datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))
    print('_Python.version:', sys.version)
    print('_Scipy.version:',scipy.__version__)
    print('_XGBoost.version:',xgb.__version__)
    print('_Sklearn.version:',sklearn.__version__)
    
def validation(filename):
        
    try:
        vib = pd.read_csv(filename, header=None, skiprows=9)
        vib = np.asarray(vib)[:,1:-1].transpose()

        meta = pd.read_csv(filename, header=None, skiprows=4, nrows=1)
        rpm = int(meta[2])

        fs = pd.read_csv(filename, header=None, skiprows=6, nrows=1)
        Fs = int(fs[1])

        TimeFeatureExtractor = Extract_Time_Features(vib)
        features_time = TimeFeatureExtractor.Features()

        FreqFeatureExtractor = Extract_Freq_Features(vib, rpm, Fs)
        features_freq = FreqFeatureExtractor.Features()

        features = np.concatenate((features_time, features_freq)).T
        data = pd.DataFrame(features)

        data.columns = ['AbsMax', 'AbsMean', 'P2P', 'RMS', 'Skewness', 'Kurtosis', 'Crest', 
                        'Shape', 'Impulse','1x', '2x', '3x', '4x', '1xB', '2xB', '3xB', '4xB']

        data['WATT'] = meta[3]
        y = pd.read_csv(filename, header=None, skiprows=3, nrows=1)
        y = y[1]

    except (NameError, IndexError, TypeError, pd.errors.EmptyDataError):
        print('Source Data Error')
        
    else:
        df = data[['WATT', 'AbsMax', 'AbsMean', 'P2P', 'RMS', 'Skewness', 'Kurtosis', 'Crest', 
                        'Shape', 'Impulse','1x', '2x', '3x', '4x', '1xB', '2xB', '3xB', '4xB']]

        norm = np.load('vibration_norm.npy')#norm
        X = (df - norm[0]) / norm[1]

        xgb_joblib = joblib.load('vibration_xgb.pkl') 
        y_pred = xgb_joblib.predict(X)
        name = os.path.basename(filename)
        print("Filename:%s, Prediction:%s, Y_Target:%d" % (name, y_pred[0], y))
        
    return y_pred[0], y[0]


start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='vibration', type=str)
args = parser.parse_args()
datapath = glob.glob(args.path + '**/**/*.csv', recursive=True)

framework()

#random = random.sample(datapath, 100)

y_hat = []
y_target = []
for file in datapath:
    y_pred, y = validation(file)
    y_hat.append(y_pred)
    y_target.append(y)
    
target_names = ['0', '1', '2', '3', '4']
print('########## 진동데이터 유효검증 결과 ##########')
print(f"검증데이터수: {len(y_hat)} 개")

accuracy = accuracy_score(y_target, y_hat)
print('예측 정확도: {0:.4f}'.format(accuracy))
print(classification_report(y_target, y_hat, target_names=target_names, digits=4)) 
print('종료시간(UCT+09:00)',datetime.datetime.now(pytz.timezone('Asia/Seoul')).strftime("%Y-%m-%d %H:%M:%S"))
end = time.time()
elapsed_time = end-start
print(f"검증소요시간：{elapsed_time}초")

y_prediction = pd.DataFrame(y_hat)
y_prediction.to_csv('y_predict_vibration.csv', index=False)