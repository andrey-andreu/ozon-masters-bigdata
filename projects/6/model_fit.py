import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_in",
    type=str
)
parser.add_argument(
    "--sklearn_model_out",
    type=str
)
args = parser.parse_args()

train = pd.read_parquet(args.train_in)
X = sp.dok_matrix((train.shape[0], train.word_vector[0]["size"]), dtype=int)
for count, value in enumerate(train.word_vector):
    X[count, value["indices"]] = 1
y = train.label

model = LogisticRegression()
model.fit(X,y)
dump(model, args.sklearn_model_out)
