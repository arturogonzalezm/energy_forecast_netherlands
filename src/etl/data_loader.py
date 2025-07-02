"""
ETL Data Loader Module
This module contains the DataLoader class, which is responsible for
reading raw CSV data, skipping headers, and parsing each line into a Spark DataFrame.
It uses a singleton SparkSession from utils.session_utils.
"""

import csv

from pyspark.sql import DataFrame
from pyspark.sql import Row

from utils.session_utils import get_spark_session


# Step 1: Define Components
class DataLoader:
    """
    Reads raw CSV, skips headers, parses each line into a Spark DataFrame.
    Uses singleton SparkSession from utils.
    """

    def __init__(self, path: str):
        self.spark = get_spark_session()
        self.path = path

    def load(self) -> DataFrame:
        raw_rdd = self.spark.sparkContext.textFile(self.path, minPartitions=1).repartition(1)
        data_rdd = (
            raw_rdd.zipWithIndex()
            .filter(lambda x: x[1] >= 3)
            .map(lambda x: x[0])
        )
        rows = (
            data_rdd.map(self._parse_line)
            .filter(lambda r: r is not None)
        )
        return self.spark.createDataFrame(rows)

    @staticmethod
    def _parse_line(line: str):
        parts = list(csv.reader([line]))[0]
        try:
            return Row(
                timestamp=parts[0],
                nuclear=float(parts[1]),
                nonrenewable=float(parts[2]),
                renewable=float(parts[3]),
                price=float(parts[4])
            )
        except ValueError:
            return None
