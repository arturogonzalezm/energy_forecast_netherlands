"""
Utility functions for managing Spark sessions in a PySpark application.
"""

from pyspark.sql import SparkSession

_spark_instance = None


def get_spark_session(app_name: str = "ElectricityForecastPipeline") -> SparkSession:
    global _spark_instance
    if _spark_instance is None:
        _spark_instance = SparkSession.builder.appName(app_name).getOrCreate()
        _spark_instance.sparkContext.setLogLevel("ERROR")
    return _spark_instance
