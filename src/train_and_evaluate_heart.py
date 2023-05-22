import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from get_data_heart import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RFC"]["params"]["n_estimators"]
    criterion = config["estimators"]["RFC"]["params"]["criterion"]
    max_depth = config["estimators"]["RFC"]["params"]["max_depth"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)
    lr = RandomForestClassifier(
        n_estimators=n_estimators, 
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    
    (n_estimators, criterion, max_depth) = eval_metrics(test_y, predicted_qualities)

    print("Random Forest model (n_estimators=%f, criterion=%f, max_depth=%f):" % (n_estimators, criterion, max_depth))
    print("  n_estimators: %s" % n_estimators)
    print("  criterion: %s" % criterion)
    print("  max_depth: %s" % max_depth)

#####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth
        }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_heart.joblib")

    joblib.dump(lr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="heart_params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)