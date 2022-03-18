### SqoopOperator Skeleton

import airflow

from airflow.models import DAG
from airflow.contrib.operators.sqoop_operator import SqoopOperator
from airflow.utils.dates import days_ago

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

# [START Dag_Sqoop_Import]
# with DAG( 
# 	  'tutorial',
#     default_args=default_args,
#     description='A simple tutorial DAG',
#     schedule_interval=timedelta(days=1),
#     start_date=datetime(2021, 1, 1),
#     catchup=False,
#     tags=['example'],
# ) as dag:
Dag_Sqoop_Import = DAG(
				
				default_args=default_args,
				dag_id="SqoopImport",
				schdule_interfal="* * * * *",
				start_date=days_ago(2)
)
# [END Dag_Sqoop_Import]

Dag_Sqoop_Import.doc_md = __doc__
Dag_Sqoop_Import.doc_md = """SqoopOperator DAG"""

# [START sqoop_config]
sqoop_config = {

			'conn_id' : 'sqoop_local',
			'cmd_type' : 'import',
    		'table' : None,
    		'query': None,
    		'target_dir' : None,
    		'append': False,
    		'file_type': 'text',
    		'columns': None,
    		'num_mappers': None,
    		'split_by': None,
    		'where': None,
    		'export_dir' : None,
    		'input_null_string' : None,
    		'input_null_non_string' : None,
    		'staging_table' : None,
    		'clear_staging_table' : False,
    		'enclosed_by' : None,
    		'escaped_by' : None,
    		'input_fields_terminated_by' : None,
    		'input_lines_terminated_by' : None,
    		'input_optionally_enclosed_by' : None,
    		'batch' : False,
    		'direct' : False,
    		'driver' : None,
    		'verbose' : False,
    		'relaxed_isolation' : False,
    		'properties' : None,
    		'hcatalog_database' : None,
    		'hcatalog_table' : None,
    		'create_hcatalog_table' : False,
    		'extra_import_options' : None,
    		'extra_export_options' : None
}
# [END sqoop_config]

# [START SparkSubmitOperator]			   
sqoop_mysql_import = SqoopOperator(
				
				task_id="sqoop_import",
				dag=Dag_Sqoop_Import,
				**sqoop_config
)
# [END SparkSubmitOperator]

sqoop_mysql_import

if __main__ == "__main__"    :
    Dag_Sqoop_Import.cli()								   