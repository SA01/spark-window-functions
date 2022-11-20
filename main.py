from pyspark import SparkConf
from pyspark.sql import Window, SparkSession, DataFrame
import pyspark.sql.functions as spark_func
import time


def create_spark_session() -> SparkSession:
    conf = SparkConf().set("spark.driver.memory", "8g")

    spark_session = SparkSession\
        .builder\
        .master("local[4]")\
        .config(conf=conf)\
        .appName("Aggregate Transform Tutorial") \
        .config("spark.jars", "postgresql-42.5.0.jar") \
        .getOrCreate()

    return spark_session


def read_data_from_db(spark: SparkSession, connection_str: str, username: str, password: str, table: str) -> DataFrame:
    properties = {
        "user": username,
        "password": password,
        "driver": "org.postgresql.Driver"
    }
    data_df = spark.read.jdbc(
        url=connection_str,
        properties=properties,
        table=table
    )

    return data_df


def read_sales(spark: SparkSession, host: str, username: str, password: str) -> DataFrame:
    table = "sales.salesorderheader"
    sales_df = read_data_from_db(spark=spark, connection_str=host, username=username, password=password, table=table)

    return sales_df


def data_with_total_sales_sale_rank(spark: SparkSession, host: str, username: str, password: str):
    # Load sales.salesorderheader data into dataframe
    raw_data = read_sales(spark=spark, host=host, username=username, password=password)

    # Specify the window. Partitioned by orderdate, ordered by totaldue in descending order
    sum_window = Window.partitionBy("orderdate")
    rank_window = Window.partitionBy("orderdate").orderBy(spark_func.col("totaldue").desc())

    # Write the query
    result = raw_data\
        .select("orderdate", "salesorderid", "totaldue")\
        .withColumn("sum_of_day", spark_func.sum("totaldue").over(sum_window))\
        .withColumn("sale_rank", spark_func.rank().over(rank_window))\
        .orderBy("orderdate", "sale_rank")

    # Display the data
    result.show(truncate=False, n=1000)


def data_with_running_average(spark: SparkSession, host: str, username: str, password: str):
    # Load sales.salesorderheader data into dataframe
    raw_data = read_sales(spark=spark, host=host, username=username, password=password).repartition(4)

    # Specify the window using partitionBy, orderBy and rowsBetween
    running_avg_window = Window.partitionBy("territoryid").orderBy("orderdate").rowsBetween(-3, -1)

    result = raw_data \
        .select("territoryid", "orderdate", "totaldue") \
        .groupby("territoryid", "orderdate")\
        .agg(spark_func.sum("totaldue").alias("totaldue"))\
        .withColumn("running_avg", spark_func.avg("totaldue").over(running_avg_window))

    result.show(truncate=False, n=1000)


if __name__ == '__main__':
    connection_str = "jdbc:postgresql://localhost:5432/Adventureworks"
    username = "postgres"
    password = "postgres"

    spark = create_spark_session()

    data_with_total_sales_sale_rank(spark=spark, host=connection_str, username=username, password=password)
    data_with_running_average(spark=spark, host=connection_str, username=username, password=password)
    time.sleep(10000)
