import os, sys
import argparse
from joblib import load
import pandas as pd
import scipy.sparse as sp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test-in",
    type=str,
    dest="test_in"
)
parser.add_argument(
    "--pred-out",
    type=str,
    dest="pred_out"
)
parser.add_argument(
    "--sklearn-model-in",
    type=str,
    dest="sklearn_model_in"
)
args = parser.parse_args()

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

# start session
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

test_path = args.test_in
pred_path = args.pred_out
model_path = args.sklearn_model_in

from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import PipelineModel

# schema = StructType([
#     StructField("id", StringType()),
#     StructField("word_vector", IntegerType()),
# ])

model = load(model_path)
test = spark.read.parquet(test_path)

X_test = sp.dok_matrix((test.shape[0], test.word_vector[0]["size"]), dtype=int)
for count, value in enumerate(test.word_vector):
    X_test[count, value.indices] = 1

pred = model.predict(X_test)
data_pred = test.select("id").toPandas()
data_pred["prediction"] = pred.astype(int)
# data_pred.to_csv(pred_path, index=False)
data_pred_spark = spark.createDataFrame(data_pred)
data_pred_spark.write.mode("overwrite").save(pred_path, header='false', format='csv')

spark.stop()
