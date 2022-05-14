import os, sys
import argparse

from pyspark.sql.types import *
from pyspark.ml.feature import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_in",
    type=str
)
parser.add_argument(
    "--path_out",
    type=str
)
args = parser.parse_args()

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/conda/envs/dsenv/bin/python"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline

conf = SparkConf()

# .master("yarn")

spark = SparkSession.builder.master("yarn").config(conf=conf).appName("Spark ML").getOrCreate()

data_path = args.path_in
save_path = args.path_out

if data_path.find("train") >= 0:
    schema = StructType([
        StructField("id", StringType()),
        StructField("label", FloatType()),
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
else:
    schema = StructType([
        StructField("id", StringType()),
    #     StructField("label", FloatType()),
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

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="word_vector", binary=True)
# assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified', 'unixReviewTime'], outputCol="features")
pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer
])

dataset = spark.read.json(data_path, schema=schema)
dataset.cache()
new_data = pipeline.fit(dataset).transform(dataset)
if "label" in new_data.columns:
    new_data2 = new_data.select("id", "word_vector", "label")
else:
    new_data2 = new_data.select("id", "word_vector")

new_data2.write.mode('overwrite').parquet(save_path)
spark.close()
