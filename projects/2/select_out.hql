INSERT OVERWRITE DIRECTORY 'andrey-andreu_hiveout'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
SELECT * FROM hw2_pred;