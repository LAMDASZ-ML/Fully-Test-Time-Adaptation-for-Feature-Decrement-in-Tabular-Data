import numpy as np
import pandas as pd
import argparse
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import tree
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import random
import itertools
from tabpfn import TabPFNClassifier

def main(dataset, task, model, impute):
    filename = './datasets/' + dataset + '.csv'
    data = pd.read_csv(filename)

    if task == "binaryclass":
        if model == "CatBoost":
            downstream = CatBoostClassifier()
        elif model == "XGBoost":
            downstream = XGBClassifier()
        elif model == "MLP":
            downstream = MLPClassifier()
        elif model == "TabPFN":
            downstream = TabPFNClassifier(device='cuda')
    elif task == "multiclass":
        if model == "CatBoost":
            downstream = CatBoostClassifier(loss_function='MultiClass')
        elif model == "XGBoost":
            downstream = XGBClassifier(eval_metric='mlogloss')
        elif model == "MLP":
            downstream = MLPClassifier()
        elif model == "TabPFN":
            downstream = TabPFNClassifier(device='cuda')
    elif task == "regression":
        if model == "CatBoost":
            downstream = CatBoostRegressor()
        elif model == "XGBoost":
            downstream = XGBRegressor()
        elif model == "MLP":
            downstream = MLPRegressor()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)
    if model == "TabPFN":
        downstream.fit(X_train_raw[:10000], y_train[:10000])
    else:
        downstream.fit(X_train_raw, y_train)

    results = []
    for num in range(0, len(data.columns) - 1):
        metric1_by_model = []
        metric2_by_model = []

        # If the number of combinations > 10000, you can use these codes.
        # combinations = []
        # while len(combinations) < 20:
        #     combination = random.sample(data.columns[:-1].tolist(), num)
        #     if combination not in combinations:
        #         combinations.append(combination)

        combinations = list(itertools.combinations(data.columns[:-1], num))
        for combination in combinations:
            X_test = X_test_raw.copy()
            for i in combination:
                if impute == "nan":
                    if model == "MLP":
                        X_test[i] = 0
                    else:
                        X_test[i] = np.nan
                elif impute == "random":
                    X_test[i] = np.random.randint(-100, 100)

            if task == "binaryclass":
                y_pred = downstream.predict(X_test)
                y_pred_proba = downstream.predict_proba(X_test)[:,1]  # 获取正类概率
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                metric1_by_model.append(accuracy)
                metric2_by_model.append(f1)

            elif task == "multiclass":
                y_pred = downstream.predict(X_test)
                y_pred_proba = downstream.predict_proba(X_test)  # 获取正类概率
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                metric1_by_model.append(accuracy)
                metric2_by_model.append(f1)

            elif task == "regression":
                y_pred = downstream.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                metric1_by_model.append(rmse)
                metric2_by_model.append(mae)

        results.append([model, num/(len(data.columns)-1), sum(metric1_by_model) / len(metric1_by_model),  sum(metric2_by_model) / len(metric2_by_model)])
    if task != "regression":
        results_df = pd.DataFrame(results, columns=["model", "%", "ACC", "F1"])
    else:
        results_df = pd.DataFrame(results, columns=["model", "%", "RMSE", "MAE"])

    results_df.to_csv("result.csv", index=False, mode='a', header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help="Dataset Name")
    parser.add_argument('--task', type=str, required=True,
                        help="Task type")
    parser.add_argument('--model', type=str, required=True,
                        help="Model Name")
    parser.add_argument('--impute', type=str, required=True,
                        help="nan or Random")
    args = parser.parse_args()
    dataset = args.dataset
    task = args.task
    model = args.model
    impute = args.impute
    main(dataset, task, model, impute)
