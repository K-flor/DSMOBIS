### SparkSubmitOperator Skeleton

import airflow
import logging
import os

from datetime import datetime, timedelta
from textwrap import dedent
from airflow.models import DAG
#from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.utils.dates import days_ago

log = logging.getLogger(__name__)

# [START default_args]
default_args = {
            
            'owner': 'airflow',
            'depends_on_past': False,
            'email': ['airflow@example.com'],
            'email_on_failure': False,
            'email_on_retry': False,
            'retries': 1,
            'retry_delay': timedelta(minutes=5),
            # 'queue': 'bash_queue',
            # 'pool': 'backfill',
            # 'priority_weight': 10,
            # 'end_date': datetime(2016, 1, 1),
            # 'wait_for_downstream': False,
            # 'dag': dag,
            # 'sla': timedelta(hours=2),
            # 'execution_timeout': timedelta(seconds=300),
            # 'on_failure_callback': some_function,
            # 'on_success_callback': some_other_function,
            # 'on_retry_callback': another_function,
            # 'sla_miss_callback': yet_another_function,
            # 'trigger_rule': 'all_success'
}
# [END default_args]

# [START instantiate_dag]
# with DAG(
#     'tutorial',
#     default_args=default_args,
#     description='A simple tutorial DAG',
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2021, 1, 1),
#     catchup=False,
#     tags=['example'],
# ) as dag:

dag_conf = DAG(
        
        dag_id="sparkSubmitOperatorDemo",
        schedule_interval="* * * * *",
        start_date=days_ago(1),
        default_args=default_args,
        description='sparkSubmitOperatorDemo sample'
)
# [END instantiate_dag]

# [START documentation]
dag_conf.doc_md = __doc__
dag_conf.doc_md = """sparkSubmitOperatorDemo DAG"""
# [END documentation]  

# [START spark_config] 
spark_config = {

           'conf' : {
                     "spark.yarn.maxAppAttempts" : 1,
                     "spark.yarn.executor.memoryOverhead" : "512"
            },
           'conn_id' : 'spark_default',
           'application': '',
           'conf' : None,
           'files' : None,
           'py_files' : None,
           'archives' : None,
           'driver_class_path' : None,
           'jars' : None,
           'java_class' : None,
           'packages' : None,
           'exclude_packages' : None,
           'repositories' : None,
           'total_executor_cores' : None,
           'executor_cores' : None,
           'executor_memory' : None,
           'driver_memory' : None,
           'keytab' : None,
           'principal' : None,
           'proxy_user' : None,
           'name' : 'airflow-spark',
           'num_executors' : None,
           'status_poll_interval' : 1,
           'application_args' : None,
           'env_vars' : None,
           'verbose' : False,
           'spark_binary' : None
}
# [END spark_config]

# [START SparkSubmitOperator]
spark_submit_local = SparkSubmitOperator(
                
                task_id="sparksubmitjobs",
                dag=dag_conf,
                **spark_config
)
# [END SparkSubmitOperator]

logging.info("SPARK_SUBMIT_OPERATOR")
logging.info("dag_conf: %s", dag_conf)
logging.info("spark_config: %s", spark_config)


spark_submit_local.doc_md = dedent(
    """
        TEST SPARK_SUBMIT_LOCAL
    """
)

spark_submit_local    

#if __main__ == "__main__"    :
#    dag_conf.cli()