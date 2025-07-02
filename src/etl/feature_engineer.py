"""
This module contains the FeatureEngineer class, which is responsible for
parsing timestamps, creating lag features, and generating time-based features
from a given DataFrame.
"""

import pyspark.sql.functions as F

from pyspark.sql import DataFrame
from pyspark.sql.window import Window


class FeatureEngineer:
    """
    Parses timestamps, creates lag features (F.lag), and time-based features.
    """

    def __init__(self, df: DataFrame):
        self.df = df

    def transform(self) -> DataFrame:
        df = self.df.withColumn(
            "timestamp", F.to_timestamp(F.col("timestamp"), "yyyy-MM-dd'T'HH:mmXXX")
        )
        window = Window.orderBy("timestamp")
        for lag_i in [1, 2, 24]:
            df = df.withColumn(f"price_lag_{lag_i}", F.lag(F.col("price"), lag_i).over(window))
        df = (
            df.withColumn("hour_of_day", F.hour(F.col("timestamp")))
            .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
        )
        return df.na.drop()
