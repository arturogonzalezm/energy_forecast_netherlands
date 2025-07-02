from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, hour, dayofweek, to_timestamp
from pyspark.sql.window import Window

# 1. Initialize Spark
spark = SparkSession.builder \
    .appName("ElectricityForecast") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")  # suppress Hadoop warnings

# 2. Read CSV, skip first three rows, single partition for zipWithIndex
raw_rdd = spark.sparkContext.textFile(
    "data/energy-charts_Electricity_production_and_spot_prices_the_Netherlands_in_week_27_2025.csv",
    minPartitions=1
).repartition(1)
data_rdd = (
    raw_rdd.zipWithIndex()
           .filter(lambda x: x[1] >= 3)
           .map(lambda x: x[0])
)

# 3. Parse lines into Rows
from pyspark.sql import Row
import csv

def parse_line(line):
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

rows = data_rdd.map(parse_line).filter(lambda r: r is not None)
df = spark.createDataFrame(rows)

# 4. Timestamp parsing and feature engineering
df = df.withColumn(
    "timestamp",
    to_timestamp(col("timestamp"), "yyyy-MM-dd'T'HH:mmXXX")
)

window = Window.orderBy("timestamp")
for lag_i in [1, 2, 24]:
    df = df.withColumn(f"price_lag_{lag_i}", lag(col("price"), lag_i).over(window))

df = df.withColumn("hour", hour(col("timestamp"))) \
       .withColumn("weekday", dayofweek(col("timestamp")))

df_fe = df.na.drop()

# 5. Move to pandas for ARIMA
pd_df = df_fe.select("timestamp", "price").toPandas().set_index("timestamp")

# 6. Fit ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(pd_df['price'], order=(1, 1, 1))
results = model.fit()

# 7. Summary and diagnostics
print(results.summary())

from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

lb_table = acorr_ljungbox(results.resid, lags=[1], return_df=True)
jb_stat, jb_pvalue, _, _ = jarque_bera(results.resid)

print("\nModel Quality Metrics:")
print(f"  AIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
print(f"  p-value AR(1): {results.pvalues.get('ar.L1', float('nan')):.3f}, MA(1): {results.pvalues.get('ma.L1', float('nan')):.3f}")
print(f"  Ljung-Box (lag 1) p-value: {lb_table['lb_pvalue'].iloc[0]:.3f}")
print(f"  Jarque-Bera p-value: {jb_pvalue:.3f}")

# 8. Forecast
forecast = results.get_forecast(steps=24)
forecast_df = forecast.summary_frame()

print("\nForecast for the next 24 hours:")
print(forecast_df)

# Interpretation:
# - 'mean': point forecast (EUR/MWh)
# - 'mean_ci_lower' / 'mean_ci_upper': 95% prediction interval
# - Wider intervals reflect growing uncertainty over time
