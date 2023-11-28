import warnings
warnings.filterwarnings('ignore')

import os
import glob
import argparse
import chardet

from multiprocessing import Pool

import numpy as np
import pandas as pd

from Current_Feature_Extractor import Extract_Time_Features, Extract_Phase_Features, Extract_Freq_Features


'''
column names of dataframe will be like this:
                    [['WATT', 'R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                   'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                   'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                   'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                   'R_Impulse', 'S_Impulse', 'T_Impulse',
                   'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                   'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                   'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']]

'''

datapath = r"D:\기계시설물 고장 예지 센서\Training\current\2.2kW\L-DSF-01\1"
savepath = '/home/gpuadmin/test_data/transformed'


# get arguments
def get_args():
    parser = argparse.ArgumentParser(description='Current Feature Extraction')
    parser.add_argument('--datapath', type=str, default=datapath)
    parser.add_argument('--savepath', type=str, default=savepath)
    args = parser.parse_args()
    return args


# make save directory if not exist
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('make directory: ', path)


# transform all files in datapath
def transform_files(datapath, savepath):
    # transform and merge all files in category-level directory
    tasks = []
    for watt in os.listdir(datapath):
        for motor in os.listdir(datapath + '/' + watt):
            for category in os.listdir(datapath + '/' + watt + '/' + motor):
                # for first file, detect encoding
                file_encoding = detect_encoding(datapath + '/' + watt + '/' + motor + '/' + category)
                category_path = datapath + '/' + watt + '/' + motor + '/' + category
                csv_name = watt + '_' + motor + '_' + category + '.csv'
                tasks.append((glob.glob(category_path + '/*.csv'), savepath, csv_name, file_encoding))

    with Pool() as p:
        p.starmap(data_transform, tasks)
        
                


# detect encoding of first file in category
def detect_encoding(category_path):
    with open(glob.glob(category_path + '/*.csv')[0], 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']
                
'''
data structure will have 3 levels of directory; watt, motor, category like this:

(data_path)
./
├── 11kW
│   ├── L-CAHU-01R
│   │   ├── 정상
│   │   ├── 축정렬불량
│   │   └── 회전체불평형
│   ├── L-CAHU-03R
│   │   └── 정상
│   ├── R-CAHU-01R
│   │   ├── 정상
│   │   └── 벨트느슨함
│   └── R-CAHU-02R
│       ├── 정상
│       └── 베어링불량
├── 15kW
│   ├── L-CAHU-01S
│   │   ├── 정상
│   │   └── 회전체불평형
│   ├── R-CAHU-01S
│   │   ├── 정상
│   │   └── 베어링불량
│   └── R-CAHU-03S
│       ├── 정상
│       └── 벨트느슨함

in the last directory, there are csv files.

after transformation, each csv file will be a single row of dataframe.
so, we're going to make a dataframe with all the data of category.
'''
def data_transform(category, savepath, csv_name, file_encoding):

    result_df = pd.DataFrame(columns=['WATT', 'R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                                    'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                                    'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                                    'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                                    'R_Impulse', 'S_Impulse', 'T_Impulse',
                                    'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                                    'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                                    'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x'])

    for filename in category:
        try:
            cur = pd.read_csv(filename, header=None, skiprows=9, encoding=file_encoding)
            cur = np.asarray(cur)[:,1:4].transpose()

            meta = pd.read_csv(filename, header=None, skiprows=4, nrows=1, encoding=file_encoding)
            rpm = int(meta[2])

            fs = pd.read_csv(filename, header=None, skiprows=6, nrows=1, encoding=file_encoding)
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
            y = pd.read_csv(filename, header=None, skiprows=3, nrows=1, encoding=file_encoding)
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


            result_df = pd.concat([result_df, df], ignore_index=True)\
    
    # print filename of category when finished
    print('category: ', csv_name, 'finished. ' + str(len(result_df)) + ' files transformed.')        

    # save result_df as csv file
    result_df.to_csv(savepath + '/' + csv_name, index=False)        
    

def main():
    args = get_args()

    datapath = args.datapath
    savepath = args.savepath

    make_dir(savepath)

    transform_files(datapath, savepath)   


if __name__ == '__main__':
    main()



    
