"""
This module orchestrates the entire electricity price forecasting pipeline.
"""

from src.models.arima_model import ARIMAModel
from src.etl.data_loader import DataLoader
from src.etl.feature_engineer import FeatureEngineer


# Step 2: Pipeline Orchestrator
class ElectricityForecastPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self):
        # Load data
        loader = DataLoader(self.data_path)
        df_raw = loader.load()

        # Engineer features
        fe = FeatureEngineer(df_raw)
        df_fe = fe.transform()

        # Prepare series for ARIMA
        pd_df = df_fe.select("timestamp", "price").toPandas().set_index("timestamp")

        # Fit and output
        model = ARIMAModel(pd_df['price'])
        results = model.fit()
        print(results.summary())

        metrics = model.diagnostics()
        print("\nModel Quality Metrics:")
        print(f"  AIC: {metrics['aic']:.2f}, BIC: {metrics['bic']:.2f}")
        print(f"  p-value AR(1): {metrics['p_ar_l1']:.3f}, MA(1): {metrics['p_ma_l1']:.3f}")
        print(f"  Ljung-Box p-value: {metrics['lb_pvalue']:.3f}")
        print(f"  Jarque-Bera p-value: {metrics['jb_pvalue']:.3f}")

        forecast_df = model.forecast()
        print("\nForecast for the next 24 hours:")
        print(forecast_df)
        return results, metrics, forecast_df
