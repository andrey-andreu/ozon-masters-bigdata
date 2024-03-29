#!/opt/conda/envs/dsenv/bin/python

from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

tokenizer = Tokenizer(inputCol="reviewText", outputCol="words")
stop_words = StopWordsRemover.loadDefaultStopWords("english")
swr = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="words_filtered", stopWords=stop_words)
count_vectorizer = CountVectorizer(inputCol=swr.getOutputCol(), outputCol="word_vector", binary=True)
assembler = VectorAssembler(inputCols=[count_vectorizer.getOutputCol(), 'verified', 'unixReviewTime'], outputCol="features")
lr = LogisticRegression(labelCol="overall", maxIter=20)
pipeline = Pipeline(stages=[
    tokenizer,
    swr,
    count_vectorizer,
    assembler,
    lr
])
