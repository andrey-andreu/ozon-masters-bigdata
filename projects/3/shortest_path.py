import os
import sys

SPARK_HOME = "/usr/hdp/current/spark2-client"
PYSPARK_PYTHON = "/opt/cloudera/parcels/CDH/lib/spark"
os.environ["PYSPARK_PYTHON"]= PYSPARK_PYTHON
os.environ["SPARK_HOME"] = SPARK_HOME

PYSPARK_HOME = os.path.join(SPARK_HOME, "python/lib")
sys.path.insert(0, os.path.join(PYSPARK_HOME, "py4j-0.10.9.3-src.zip"))
sys.path.insert(0, os.path.join(PYSPARK_HOME, "pyspark.zip"))

from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f

conf = SparkConf()

spark = SparkSession.builder.config(conf=conf).appName("Spark SQL").getOrCreate()

schema = StructType(fields=[
    StructField("column_1", IntegerType()),
    StructField("column_0", IntegerType())
])

start = int(sys.argv[1])
end = int(sys.argv[2])
input_file  = sys.argv[3]
output_file = sys.argv[4]

df = spark.read.csv(input_file, schema=schema, sep=r"\t")
df = df.distinct().cache()

temporary_data = df.filter(f'column_0 = {start}')
current_len = 1
while True:
    df = df.withColumnRenamed(f'column_{current_len}', f'column_{current_len+1}').withColumnRenamed(f'column_{current_len-1}', f'column_{current_len}')
    temporary_data = temporary_data.join(df, f'column_{current_len}').cache()
    if temporary_data.filter(f'column_{current_len+1} = {end}').count() > 0:
        break
    current_len += 1
columns = ["column_" + str(i) for i in range(current_len+2)]
# answer_arr = temporary_data.filter(f'column_{current_len+1} = {end}')[columns].collect()
temporary_data.filter(f'column_{current_len+1} = {end}')[columns].write.csv(path=output_file, mode='overwrite')
# answer = [list(an) for an in answer_arr]
# ans = spark.createDataFrame(answer)
# ans.write.csv(path=output_file, mode='overwrite')
