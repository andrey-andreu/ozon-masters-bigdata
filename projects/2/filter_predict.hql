ADD FILE 2.joblib;
ADD FILE projects/2/model.py;
ADD FILE projects/2/predict.py;

INSERT OVERWRITE TABLE hw2_pred
SELECT TRANSFORM(*)
USING 'predict.py' FROM hw2_test WHERE if1 > 20 and if1 < 40;
