"""
Main entry point for the electricity forecasting pipeline.
"""
from src.pipeline import ElectricityForecastPipeline

if __name__ == '__main__':
    pipeline = ElectricityForecastPipeline(
        "data/energy-charts_Electricity_production_and_spot_prices_the_Netherlands_in_week_27_2025.csv"
    )
    pipeline.run()
