#!/opt/conda/envs/dsenv/bin/python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('WARN')

from pyspark.ml import PipelineModel
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.ml.feature import *

import sys

model_path = sys.argv[0]
test_path = sys.argv[1]
result_path = sys.argv[2]

schema = StructType([
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

model = PipelineModel.load(model_path)
test = spark.read.json(test_path, schema=schema)

test = test.withColumn("vote", test["vote"].cast(IntegerType()))
test = test.na.fill(value=0)
# test = test.na.fill("NaN")
# # test = test.withColumn('review', f.concat_ws(' ', f.col("reviewText"), f.col("summary")))

# test = test.select(
#  'vote',
#  'verified',
#  'reviewText',
#  'unixReviewTime').cache()

pred = model.transform(test)
pred.write().overwrite().save(result_path)