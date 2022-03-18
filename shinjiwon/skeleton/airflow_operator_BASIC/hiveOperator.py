### HiveOperator Skeleton

import airflow
import logging
import os

from textwrap import dedent
from airflow.models import DAG
from airflow.operators.hive_operator import HiveOperator
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

# [START dag_hive]
# with DAG(
#     'tutorial',
#     default_args=default_args,
#     description='A simple tutorial DAG',
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2021, 1, 1),
#     catchup=False,
#     tags=['example'],
# ) as dag:

dag_hive = DAG(
        
        default_args=default_args,
        dag_id='hiveOperator',
        schedule_interval='* * * * *',
        start_date=days_ago(1)
)
# [END dag_hive]

dag_hive.doc_md = __doc__
dag_hive.doc_md = """hiveOperator DAG"""
               
hql_query = """use training;
create table if not exists airflow_hive (id int, name string);
insert into airflow_hive values(1,"bigdata");"""

# [START hive_config] 
hive_config = {

          'hql' : 'hql_query',
          'hive_cli_conn_id' : 'hive_local',
          'schema' : 'default',
          'hiveconfs' : None,
          'hiveconf_jinja_translate': False,
          ' script_begin_tag' : None,
          'run_as_owner': False,
          'mapred_queue' : None,
          'mapred_queue_priority' : None,
          'mapred_job_name' : None
}
# [END hive_config]

# [START HiveOperator]
hive_task = HiveOperator(

          task_id='hive_script_task',
          dag=dag_hive,
          **hive_config
)
# [END HiveOperator]

hive_task.doc_md = dedent(
  """ 
    TEST HIVE_TASK
  """
)

logging.info("HIVE_OPERATOR")
logging.getLogger(dag_hive)
logging.getLogger(hive_task)

hive_task

if __name__ == "__main__":
  dag_hive.cli()