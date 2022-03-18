from builtin import range
from datetime import timedelta

from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago

# [START args]
args = {
    
    'owner' : 'Airflow',
    'start_date' : days_ago(2),
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
# [END args]

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
dag = DAG(
    
    dag_id = 'example_bash_operator',
    default_args = args,
    schedule_interval = '0 0 * * *',
    dagrun_timeout=timedelta(minutes = 60),
    tags = ['example']
)
# [END instantiate_dag]

# [START howto_operator_bash]
run_this = BashOperator(
    
    task_id = 'run_after_loop',
    bash_command = 'echo 1',
    dag = dag
)

run_this_last = DummyOperator(
    
    task_id = 'run_this_last',
    dag = dag
)    
# [START howto_operator_bash]

run_this >> run_this_last

for i in range(3):
    task = BashOperator(
        
        task_id = 'runme_' + str(i)
        bash_command = 'echo "{{ task_instance_key_str }}" && sleep 1',
        dag = dag
    )
    
    task >> run_this
    
also_run_this = BashOperator(
    
    task_id = 'also_run_this'
    bash_command = 'echo "run_id = {{ run_id }} | dag_run = {{ dag_run }}"',
    dag = dag
)

also_run_this >> run_this_last   

if __name__ == "__main__":
    dag.cli() 

