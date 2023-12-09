import pickle

import joblib
import lightgbm as lgb
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression

# 데이터 불러오기 (예시)
# X, y = 데이터 불러오기

# 모델 불러오기 및 SHAP 값 계산
folders = ["current_model", "vibrate_model"]
model_types = ["logit", "lgb", "xgb"]
models = []
model = joblib.load('current_model/xgb_model1.pkl')
#
# for folder in folders:
#     for model_type in model_types:
#         for i in range(1, 5):
#             model_path = f'{folder}/{model_type}_model{i}.json'
#             print(model_path)
#             if model_type == "logit":
#                 model = pickle.load(open(model_path, 'rb'))
#
#             elif model_type == "lgb":
#                 model = lgb.Booster(model_file=model_path)
#
#             elif model_type == "xgb":
#                 model = joblib.load('current_model/xgb_model1.pkl')
#             models.append(model)

# # 각 모델에 대해 SHAP 값 계산 및 플롯
# for model in models:
#     explainer = shap.Explainer(model, X)
#     shap_values = explainer(X)
#
#     # SHAP 요약 플롯 그리기
#     shap.summary_plot(shap_values, X)
