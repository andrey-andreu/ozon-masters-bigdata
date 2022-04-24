#!/opt/conda/envs/dsenv/bin/python

import os, sys
import logging
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from joblib import dump

import mlflow

#
# Import model definition
#
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_path', type=str)
    parser.add_argument('model_param1', type=int, default=200)
    parse_args = parser.parse_args()
    from model import model, fields

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
    df = pd.read_table(parse_args.train_path, **read_table_opts)

    #split train/test
    X_train = df.iloc[:,2:15]
    y_train = df.iloc[:,1]

    #
    # Train the model
    #
    mlflow.sklearn.autolog()
    with mlflow.start_run():
        model.set_params(logisticregression__n_estimators=parse_args.model_param1)
        model.fit(X_train, y_train)


    # model.fit(X_train, y_train)

    # y_pred = model.predict_proba(X_test)

    # model_score = log_loss(y_test, y_pred[:, 1])

    # logging.info(f"model score: {model_score:.3f}")

    # # save the model
    # dump(model, "{}.joblib".format(proj_id))

if __name__ == "__main__":
    main()