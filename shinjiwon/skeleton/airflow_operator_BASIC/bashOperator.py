# Operators -> bash operators (shell commands),
#              dummy operators (grouping task based on DAG id)
# scheduling -> 1 min interval as per CRON TAB

import airflow

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago

dag_basic = DAG(
    
    dag_id = "bashOperatorDemo",
    description= "Execute Shell Command"
    schedule_interval = "* * * * *",
    start_date = days_ago(2)
)

task_2 = DummyOperator(
    
    task_id = "GroupingDag", 
    dag=dag_basic
)

task_1 = BashOperator(
    
    task_id = "Shell_execute", 
    bash_cmd = "echo 'welcome its joke' >> /usr/demo/airflow.txt", 
    dag= dag_basic
)

task_1.set_upstream(task_2)