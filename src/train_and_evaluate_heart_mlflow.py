import yaml
import pandas as pd
import argparse
from pkgutil import get_data
from get_data_heart import get_data, read_params
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import metrics
import joblib
import json
import numpy as np
import os
import mlflow
import mlflow.sklearn
import logging
from urllib.parse import urlparse
from mlflow.models.signature import infer_signature

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = metrics.mean_absolute_error(actual,pred)
    mse = metrics.mean_squared_error(actual, pred)
    #score1 = lr.score(train_x, test_x)
    return rmse, mae, mse



def train_and_evaluate_mlflow(config_path):
    config=read_params(config_path)
    test_data_path=config["split_data"]["test_path"]
    train_data_path=config["split_data"]["train_path"]
    random_state=config["base"]["random_state"]
    model_dir=config["model_dir"]

    n_estimators = config["estimators"]["RFC"]["params"]["n_estimators"]
    criterion = config["estimators"]["RFC"]["params"]["criterion"]
    max_depth = config["estimators"]["RFC"]["params"]["max_depth"]

    target=config["base"]["target_col"]
    train=pd.read_csv(train_data_path, sep=",")
    test=pd.read_csv(test_data_path, sep=",")

    train_x=train.drop(target, axis=1)
    test_x=test.drop(target, axis=1)

    train_y=train[target]
    test_y=test[target]
    

    ######################################
    mlflow_config = config["mlflow_config"]
    remote_server_uri= mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        lr = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, random_state=random_state)
        lr.fit(train_x, train_y)
        predicted_qualities=lr.predict(test_x)
        (rmse, mae, mse) = eval_metrics(test_y, predicted_qualities)
        #signature = infer_signature(train_x, predicted_qualities)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        tracking_url_type_store=urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store !="file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr, "model")        
        
        #os.makedirs(model_dir, exist_ok=True)
        #model_path = os.path.join(model_dir, "model_heart.joblib")

        #joblib.dump(lr, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="heart_params.yaml")
    parsed_args=args.parse_args()
    train_and_evaluate_mlflow(config_path=parsed_args.config)