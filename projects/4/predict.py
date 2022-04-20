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

model = PipelineModel.load(model_path)
test = spark.read.json(test_path, schema=schema, multiLine=True)

test = test.withColumn("vote", test["vote"].cast(IntegerType()))
test = test.na.fill(value=0)
test = test.na.fill("NaN")
test = test.withColumn('review', f.concat_ws(' ', f.col("reviewText"), f.col("summary")))

test = test.select(
 'overall',
 'vote',
 'verified',
 'review',
 'unixReviewTime').cache()

prediction = model.transform(test)
prediction.to_csv(result_path, header=None, index=False)