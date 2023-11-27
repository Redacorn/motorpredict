import os
import xgboost as xgb
import pandas as pd
import numpy as np

norm_data_path = r"D:\기계시설물 고장 예지 센서\Training\current\2.2kW\L-DSF-01\transform_norm"
fail_data_path = r"D:\기계시설물 고장 예지 센서\Training\current\2.2kW\L-DSF-01\transform_fail"

# read data
def read_data(path):
    '''
      set column names with ['R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                'R_Impulse', 'S_Impulse', 'T_Impulse',
                'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x']
                '''
    
    # make dataframe that has column names
    data = pd.DataFrame(columns=['R_AbsMax', 'S_AbsMax', 'T_AbsMax', 'R_AbsMean', 'S_AbsMean','T_AbsMean',
                'R_P2P', 'S_P2P', 'T_P2P', 'R_RMS', 'S_RMS', 'T_RMS', 
                'R_Skewness', 'S_Skewness', 'T_Skewness', 'R_Kurtosis', 'S_Kurtosis', 'T_Kurtosis',
                'R_Crest', 'S_Crest', 'T_Crest', 'R_Shape', 'S_Shape', 'T_Shape',
                'R_Impulse', 'S_Impulse', 'T_Impulse',
                'RS_phase', 'ST_phase', 'TR_phase', 'RS_Level', 'ST_Level', 'TR_Level',
                'R_1x', 'S_1x', 'T_1x', 'R_2x', 'S_2x', 'T_2x',
                'R_3x', 'S_3x', 'T_3x', 'R_4x', 'S_4x', 'T_4x'])
    
    # get list of files in directory
    files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

    # append each file's row to data
    for file in files:
        df = pd.read_csv(file)
        # concat data and df
        data = pd.concat([data, df], ignore_index=True)

    return data

def main():
    # read data
    norm_data = read_data(norm_data_path)
    fail_data = read_data(fail_data_path)

    # set label
    norm_data['label'] = 0
    fail_data['label'] = 1

    # concat norm_data and fail_data
    data = pd.concat([norm_data, fail_data], ignore_index=True)

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # split data into X and y
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # split data into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluate
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"accuracy: {accuracy}")
    print(f"f1 score: {f1}")


if __name__ == "__main__":
    main()