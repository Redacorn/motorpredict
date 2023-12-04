
import os
import glob
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.metrics import accuracy_score, classification_report
from Current_Feature_Extractor import Extract_Time_Features, Extract_Phase_Features, Extract_Freq_Features
import re


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

# split data into train and test

def train(df):
    # split data into X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # train model with GPU
    model = xgb.XGBClassifier(
        tree_method='gpu_hist',  # Use GPU accelerated algorithm
        # 다음 매개변수는 필요에 따라 조정할 수 있음
        gpu_id=0,               # GPU ID, 멀티 GPU 시스템에서 선택적 사용

        predictor='gpu_predictor' # 예측에 GPU를 사용
    )
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

    return model

def explain_with_shap(model, X_train, X_test):
    # SHAP 값을 계산
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    plt.subplots_adjust(left=0.4)
    # SHAP 요약 플롯 출력
    shap.summary_plot(shap_values, X_test)

    return shap_values

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def train_surrogate_model(original_model, X_train, y_train):
    # 서로게이트 모델로 결정 트리 사용
    surrogate = DecisionTreeClassifier(max_depth=3)  # 깊이는 필요에 따라 조정
    surrogate.fit(X_train, original_model.predict(X_train))
    return surrogate

def plot_surrogate_model(surrogate, feature_names):
    # 서로게이트 모델 시각화
    plt.figure(figsize=(20,10))
    plot_tree(surrogate, feature_names=feature_names, filled=True, rounded=True)
    plt.show()

def main():
    path = '../transformed/current'
    abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), path)
    folder_path = abs_path

    datapath = glob.glob(os.path.join(folder_path, '*'))
    df_list = []
    df_list = df_preprocess(datapath) #state 별 df 병합

    # train model
    train_df = merge_and_label_dfs(df_list, 1)
    model_1 = train(train_df)
    train_df = merge_and_label_dfs(df_list, 2)
    model_2 = train(train_df)
    train_df = merge_and_label_dfs(df_list, 3)
    model_3 = train(train_df)
    train_df = merge_and_label_dfs(df_list, 4)
    model_4 = train(train_df)

    # 각 모델에 대한 SHAP과 서로게이트 모델 생성 및 시각화
    for i, model in enumerate([model_1, model_2, model_3, model_4], start=1):
        # 데이터 분할
        train_df = merge_and_label_dfs(df_list, i)
        X_train, X_test, y_train, y_test = train_test_split(train_df.iloc[:, :-1], train_df.iloc[:, -1], random_state=1        )

        # SHAP 해석
        shap_values = explain_with_shap(model, X_train, X_test)

        # 서로게이트 모델 훈련 및 시각화
        #surrogate = train_surrogate_model(model, X_train, y_train)

        #plot_surrogate_model(surrogate, X_train.columns)


if __name__ == "__main__":
    main()
