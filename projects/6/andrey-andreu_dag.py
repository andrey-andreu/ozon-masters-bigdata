from datetime import datetime, timedelta
from textwrap import dedent

# The DAG object; we'll need this to instantiate a DAG
from airflow import DAG

# Operators; we need this to operate!
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
with DAG(
    'andrey-andreu_dag',
    schedule_interval=None,
    start_date=datetime(2022, 5, 14),
    catchup=False,
) as dag:
    
    base_dir = '{{ dag_run.conf["base_dir"] if dag_run else "" }}'
    t1 = SparkSubmitOperator(
        task_id='feature_eng_train_task',
        application=f"{base_dir}/feat_eng.py",
        application_args = ["--path_in", "/datasets/amazon/all_reviews_5_core_train_extra_small_sentiment.json",
                           "--path_out", "andrey-andreu_train_out"],
        spark_binary="/usr/bin/spark-submit", 
        env_vars={"PYSPARK_PYTHON": "/opt/conda/envs/dsenv/bin/python2"}
    )
    
    t2 = SparkSubmitOperator(
        task_id='feature_eng_test_task',
        application=f"{base_dir}/feat_eng.py",
        application_args = ["--path_in", "/datasets/amazon/all_reviews_5_core_test_extra_small_features.json",
                           "--path_out", "andrey-andreu_test_out"],
        spark_binary="/usr/bin/spark-submit", 
        env_vars={"PYSPARK_PYTHON": "/opt/conda/envs/dsenv/bin/python2"}
    )

    t3 = BashOperator(
        task_id='download_train_task',
        bash_command=f'hdfs dfs -get andrey-andreu_train_out {base_dir}/andrey-andreu_train_out_local'
    )
    
    t4 = BashOperator(
        task_id='train_task',
        bash_command=f'python {base_dir}/feat_eng.py --train_in {base_dir}/nick_train_out_local --sklearn_model_out {base_dir}6.joblib'
    )

    t5 = FileSensor(
        task_id='model_sensor',
        filepath=f'{base_dir}/6.joblib'
    )
    
    t6 = SparkSubmitOperator(
        task_id='predict_task',
        application=f"{base_dir}/feat_test.py",
        application_args = ["--test-in", 'hdfs:///user/andrey-andreu/andrey-andreu_test_out',
                           "--pred-out", "andrey-andreu_hw6_prediction",
                           "--sklearn-model-in", f'{base_dir}/6.joblib'],
        spark_binary="/usr/bin/spark-submit", 
        env_vars={"PYSPARK_PYTHON": "/opt/conda/envs/dsenv/bin/python2"}
    )
    
    t1 >> t2 >> t3 >> t4 >> t5 >> t6 