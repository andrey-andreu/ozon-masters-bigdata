#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging
import argparse

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump
from model import model, fields

import mlflow

#
# Import model definition
#
def main():
    # numeric_features = ["if"+str(i) for i in range(1,14)]
    # categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]

    # fields = ["id", "label"] + numeric_features + categorical_features
    # parser = argparse.ArgumentParser()

    # parser.add_argument('train_path', type=str)
    # parser.add_argument('model_param1', type=int, default=200)
    # parse_args = parser.parse_args()

    try: 
      train_path = sys.argv[1] 
      model_param1 = int(sys.argv[2]) 
    except: 
        sys.exit(1)

    #
    # Logging initialization
    #
    # logging.basicConfig(level=logging.DEBUG)
    # logging.info("CURRENT_DIR {}".format(os.getcwd()))
    # logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
    # logging.info("ARGS {}".format(sys.argv[1:]))

    #
    # Read script arguments
    #
    # try:
    #   proj_id = sys.argv[1] 
    #   train_path = sys.argv[2]
    # except:
    #   logging.critical("Need to pass both project_id and train dataset path")
    #   sys.exit(1)


    # logging.info(f"TRAIN_ID {proj_id}")
    # logging.info(f"TRAIN_PATH {train_path}")

    #
    # Read dataset
    #
    #fields = """doc_id,hotel_name,hotel_url,street,city,state,country,zip,class,price,
    #num_reviews,CLEANLINESS,ROOM,SERVICE,LOCATION,VALUE,COMFORT,overall_ratingsource""".replace("\n",'').split(",")

    read_table_opts = dict(sep="\t", names=fields, index_col=False)
    df = pd.read_table(train_path, **read_table_opts)
    # #split train/test
    # X_train, X_test, y_train, y_test = train_test_split(
    # df.iloc[:,2:15], df.iloc[:,1], test_size=0.33, random_state=42
    # )
    # df = df.astype(int)
    X = df.iloc[:,2:15]
    y = df.iloc[:,1]

    for col in X:
        X.loc[:, col] = np.floor(pd.to_numeric(X.loc[:, col], errors='coerce')).astype('Int64')
    #
    # Train the model
    #
    # model2 = Pipeline(steps=[
    # # ('preprocessor', preprocessor),
    # ('gradboosting', LogisticRegression(random_state=0, max_iter=100))
    # ])
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model.set_params(gradboosting__max_iter=model_param1)
        model.fit(X, y)
        
        # #log model params
        # mlflow.log_param("model_param1", model_param1)
        mlflow.sklearn.log_model(model, artifact_path="model_5mla")
        
        # pred = model.predict(X_test)
        # model_score = log_loss(y_test, pred)
        # mlflow.log_metrics({"log_loss": model_score})


    # model.fit(X_train, y_train)

    # y_pred = model.predict_proba(X_test)

    # model_score = log_loss(y_test, y_pred[:, 1])

    # logging.info(f"model score: {model_score:.3f}")

    # # save the model
    # dump(model, "{}.joblib".format(proj_id))

if __name__ == "__main__":
    main()
