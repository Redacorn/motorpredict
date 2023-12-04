# global
import os
import re
import glob
import argparse

# data processing
import numpy as np
import pandas as pd

# train data
from sklearn.model_selection import train_test_split

# models
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# evaluation
from sklearn.metrics import accuracy_score, f1_score


def arg_parse():
    # params : data_path, save_path
    parser = argparse.ArgumentParser(description='usage: python train_current.py --data_path <data_path> --save_path <save_path>')
    parser.add_argument('--data_path', type=str, default='/home/gpuadmin/test_data/transformed/current', help='data path')
    parser.add_argument('--save_path', type=str, default='../model', help='model save path')

    # if no ../model folder, make folder
    if not os.path.exists('../model'):
        os.makedirs('../model')
    
    args = parser.parse_args()
    return args


def state(filename):
    # 한글 유니코드 범위를 사용하여 정규 표현식 패턴을 정의
    text = os.path.basename(filename)
    korean_pattern = re.compile('[\uAC00-\uD7AF]+')

    # 주어진 텍스트에서 한글 부분만 찾아내어 리스트로 반환
    res = korean_pattern.findall(text)[0]
    state=-1
    if res == "정상":
        state=0
    elif res == "베어링불량":
        state=1
    elif res == "회전체불평형":
        state=2
    elif res == "축정렬불량":
        state=3
    elif res == "벨트느슨함":
        state=4

    return state


def df_preprocess(datapath):
    df_0 = pd.DataFrame()
    df_1 = pd.DataFrame()
    df_2 = pd.DataFrame()
    df_3 = pd.DataFrame()
    df_4 = pd.DataFrame()

    for file in datapath:
        print(file, state(file))
        data = pd.read_csv(file)
        # add new column 'state'
        if state(file) == 0:
            data['state'] = 0
            df_0 = df_0.append(data)
        elif state(file) == 1:
            data['state'] = 1
            df_1 = df_1.append(data)
        elif state(file) == 2:
            data['state'] = 2
            df_2 = df_2.append(data)
        elif state(file) == 3:
            data['state'] = 3
            df_3 = df_3.append(data)
        elif state(file) == 4:
            data['state'] = 4
            df_4 = df_4.append(data)
    return df_0, df_1, df_2, df_3, df_4


def merge_and_label_dfs(df_list, target_state):
    # 결과 데이터프레임을 저장할 리스트
    processed_dfs = []

    for state, df in enumerate(df_list):
        # 대상 상태에 해당하는 경우 'state'를 1로 설정
        if state == target_state:
            df['state'] = 1
        else:
            # 그 외의 경우 'state'를 0으로 설정
            df['state'] = 0

        # 처리된 데이터프레임을 리스트에 추가
        processed_dfs.append(df)

    # 모든 데이터프레임을 하나로 합침
    merged_df = pd.concat(processed_dfs, ignore_index=True)

    return merged_df


def pred_and_eval(model, X_test, y_test):
    # predict
    y_pred = model.predict(X_test)

    # evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"accuracy: {accuracy}")
    print(f"f1 score: {f1}")

    return accuracy, f1


def train(df, save_path):
    # train 3 models; xgboost, lightgbm, logistic regression
    # split data into X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # train model with GPU
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist',  # Use GPU accelerated algorithm
        # 다음 매개변수는 필요에 따라 조정할 수 있음
        gpu_id=0,               # GPU ID, 멀티 GPU 시스템에서 선택적 사용
        n_gpus=-1,              # 모든 GPU를 사용
        predictor='gpu_predictor' # 예측에 GPU를 사용
    )
    xgb_model.fit(X_train, y_train)
    # save xgb model
    xgb_model.save_model(os.path.join(save_path, 'xgb_model.json'))
    
    lgb_model = lgb.LGBMClassifier(
        device='gpu',           # Use GPU acceleration
        gpu_platform_id=0,      # platform id
        gpu_device_id=0         # device id
    )
    lgb_model.fit(X_train, y_train)
    # save lgb model
    lgb_model.booster_.save_model(os.path.join(save_path, 'lgb_model.json'))

    logit_model = LogisticRegression()
    logit_model.fit(X_train, y_train)
    # save logit model
    logit_model.save_model(os.path.join(save_path, 'logit_model.json'))

    # predict and evaluate
    xgb_result = pred_and_eval(xgb_model, X_test, y_test)
    lgb_result = pred_and_eval(lgb_model, X_test, y_test)
    logit_result = pred_and_eval(logit_model, X_test, y_test)

    # save result
    result_df = pd.DataFrame([xgb_result, lgb_result, logit_result], columns=['accuracy', 'f1 score'], index=['xgb', 'lgb', 'logit'])
    result_df.to_csv(os.path.join(save_path, 'result.csv'))

    return xgb_model, lgb_model, logit_model


def main():

    # parse arg
    args = arg_parse()
    data_path = args.data_path
    save_path = args.save_path
    
    path = data_path
    abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), path)
    folder_path = abs_path

    datapath = glob.glob(os.path.join(folder_path, '*'))
    df_list = []
    df_list = df_preprocess(datapath) #state 별 df 병합

    # train model
    train_df = merge_and_label_dfs(df_list, 1)
    model_1 = train(train_df, save_path)
    train_df = merge_and_label_dfs(df_list, 2)
    model_2 = train(train_df, save_path)
    train_df = merge_and_label_dfs(df_list, 3)
    model_3 = train(train_df, save_path)
    train_df = merge_and_label_dfs(df_list, 4)
    model_4 = train(train_df, save_path)


if __name__ == "__main__":
    main()
