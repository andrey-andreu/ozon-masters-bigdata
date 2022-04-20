import os
import sys

from pyspark.sql import SparkSession

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.7-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from model import pipeline

from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.ml.feature import *

train_path = sys.argv[1]
model_path = sys.argv[2]

schema = StructType([
    StructField("overall", FloatType()),
    StructField("vote", StringType()),
    StructField("verified", BooleanType()),
    StructField("reviewTime", StringType()),
    StructField("reviewerID", StringType()),
    StructField("asin", StringType()),
    StructField("reviewerName", StringType()),
    StructField("reviewText", StringType()),
    StructField("summary", StringType()),
    StructField("unixReviewTime", IntegerType()),
])

train = spark.read.json(train_path, schema=schema)

train = train.withColumn("vote", train["vote"].cast(IntegerType()))
train = train.na.fill(value=0)
train = train.na.fill("NaN")
train = train.withColumn('review', f.concat_ws(' ', f.col("reviewText"), f.col("summary")))

train = train.select(
 'overall',
 'vote',
 'verified',
 'review',
 'unixReviewTime').cache()

pipeline_model = pipeline.fit(train)
pipeline_model.write().overwrite().save(model_path)
