from pyspark.sql.session import SparkSession
import pyspark.sql.functions as file

from pyspark import SparkContext
sc = sparkContext("local", "StreamTest")

spark = SparkSession.builder.appName("BasicSparkSubmitDemo").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

mySqlRead = spark.read.format("jdbc")\
                  .option("url","jdbc:mysql://localhost:3306/bigdatapedia?useSSL=true)\
                  .option("driver","com.mysql.jdbc.Driver")\
                  .option("dbtable","customer")\
                  .option("user","root")\
                  .option("password","root").load()\

mySqlRead.write.format("jdbc")\
              .option("url","jdbc:mysql://localhost:3306/bigdatapedia?useSSL=true)\
              .option("driver","com.mysql.jdbc.Driver")\
              .option("dbtable","customer_test")\
              .option("user","root")\
              .option("password","root").mode("append").save()              
                       
