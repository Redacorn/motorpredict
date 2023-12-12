# global
import os
import glob
import argparse

# multiprocessing
import multiprocessing

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
import matplotlib.pyplot as plt
import shap

from tqdm import tqdm

def arg_parse():
    # params : data_path, save_path
    parser = argparse.ArgumentParser(description='usage: python train_current_xai.py --data_path <data_path> --save_path <save_path>')
    parser.add_argument('--data_path', type=str, default='/home/gpuadmin/motorpredict/data', help='data path')
    parser.add_argument('--save_path', type=str, default='../model', help='model save path')
    parser.add_argument('--result_path', type=str, default='../result', help='result save path')

    # if no ../model folder, make folder
    if not os.path.exists('../model'):
        os.makedirs('../model')

    if not os.path.exists('../result'):
        os.makedirs('../result')
    
    args = parser.parse_args()
    return args


# state = 0: 정상
# state = 1: 베어링 불량
# state = 2: 회전체 불평형
# state = 3: 축 정렬 불량
# state = 4: 벨트 느슨함
  

def pred_and_eval(model, X_test, y_test):
    # predict
    y_pred = model.predict(X_test)

    # evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"accuracy: {accuracy}")
    print(f"f1 score: {f1}")

    return accuracy, f1


def train(df, save_path, model_num, result_path):
    # train 3 models; xgboost, lightgbm, logistic regression
    # split data into X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2, train_size=0.8)

    # print head of x_train and y_train
    # print(X_train.head())
    # print(y_train.head())

    # train model with GPU
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist',  # Use GPU accelerated algorithm
        # 다음 매개변수는 필요에 따라 조정할 수 있음
        gpu_id=0,               # GPU ID, 멀티 GPU 시스템에서 선택적 사용
        n_gpus=-1,              # 모든 GPU를 사용
        predictor='gpu_predictor' # 예측에 GPU를 사용
    )
    xgb_model.fit(X_train, y_train)
  
    # save xgb model as pkl
    # pickle.dump(xgb_model, open(os.path.join(save_path, 'xgb_model' +  str(model_num) + '.pkl'), 'wb'))
    
    lgb_model = lgb.LGBMClassifier(
        device='gpu',           # Use GPU acceleration
        gpu_platform_id=0,      # platform id
        gpu_device_id=0         # device id
    )
    lgb_model.fit(X_train, y_train)
    # save lgb model
    # lgb_model.booster_.save_model(os.path.join(save_path, 'lgb_model' + str(model_num) + '.json'))

    logit_model = LogisticRegression(max_iter=1000)
    logit_model.fit(X_train, y_train)

    # save logit model
    # joblib.dump(logit_model, os.path.join(save_path, 'logit_model'  + str(model_num) + '.json'))

    # predict and evaluate
    xgb_result = pred_and_eval(xgb_model, X_test, y_test)
    lgb_result = pred_and_eval(lgb_model, X_test, y_test)
    logit_result = pred_and_eval(logit_model, X_test, y_test)

    num_processes = multiprocessing.cpu_count()

    # shap value
    shap_values = parallel_shap_analysis((xgb_model, X, y, 'xgb {}'.format(model_num), result_path), num_processes=num_processes)
    shap_values = parallel_shap_analysis((lgb_model, X, y, 'lgb {}'.format(model_num), result_path), num_processes=num_processes)

    # coef value plot
    print(logit_model.coef_)
    plt.plot(logit_model.coef_)
    plt.title('coef value {}'.format(model_num))
    plt.savefig(result_path + '/coef value {}.jpg'.format(model_num))
  
    # save result
    result_df = pd.DataFrame([xgb_result, lgb_result, logit_result], columns=['accuracy', 'f1 score'], index=['xgb', 'lgb', 'logit'])
    result_df.to_csv(os.path.join(save_path, 'result' + str(model_num) + '.csv'))

    return xgb_model, lgb_model, logit_model


def worker(input_data):
    # shap_analysis 함수를 호출하고 결과를 반환합니다.
    model, X, y, model_name, result_path = input_data
    return shap_analysis(model, X, y, model_name, result_path)


def parallel_shap_analysis(input_data, num_processes):
    # 입력 데이터를 균일한 형태의 배열로 변환합니다.
    input_data = np.array([list(x) for x in input_data])

    # 입력 데이터를 프로세스 수에 따라 분할합니다.
    split_input_data = np.array_split(input_data, num_processes)

    with multiprocessing.Pool(num_processes) as pool:
        # 각 프로세스에서 worker 함수를 실행합니다.
        results = pool.map(worker, split_input_data)

    # 결과를 반환합니다.
    return results


# shap 분석
def shap_analysis(model, X, y, model_name, result_path):
    # SHAP Explainer 생성

    print('shap analysis of' + model_name)
    explainer = shap.Explainer(model, X)
  
    # SHAP 값 계산
    shap_values = explainer(X)

    # 시각화: SHAP 요약 플롯 및 jpg 저장
    shap.summary_plot(shap_values, X, plot_type="bar")
    plt.title('shap summary plot({})'.format(model_name))
    plt.savefig(result_path + '/shap_summary_plot({}).jpg'.format(model_name))

    return shap_values


def main():

    # parse arg
    args = arg_parse()
    data_path = args.data_path
    save_path = args.save_path
    result_path = args.result_path

    # train_df 불러오기
    df_list, x_list, y_list = [], [], []

    for i in tqdm(range(5)):
        df = pd.read_csv(data_path + '/train_df_{}.csv'.format(i))
        df_list.append(df.iloc[:, 1:])
        # x_list.append(df.drop(['state'], axis=1).drop(['Unnamed: 0'], axis=1))
        # y_list.append(df['state'])

    # train model
    model_0 = train(df_list[0], save_path, 0, result_path)
    model_1 = train(df_list[1], save_path, 1, result_path)
    model_2 = train(df_list[2], save_path, 2, result_path)
    model_3 = train(df_list[3], save_path, 3, result_path)
    model_4 = train(df_list[4], save_path, 4, result_path)


if __name__ == "__main__":
    main()
