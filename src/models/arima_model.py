"""
This module implements an ARIMA model using statsmodels.
"""

import warnings
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera

# Suppress statsmodels frequency warning
warnings.filterwarnings("ignore", message="No frequency information.*")


class ARIMAModel:
    """
    Fits ARIMA via statsmodels; for a pure Spark approach consider spark-ts or mapInPandas.
    """

    def __init__(self, series: pd.Series, order=(1, 1, 1)):
        self.series = series
        self.order = order
        self.results = None

    def fit(self):
        model = ARIMA(self.series, order=self.order)
        self.results = model.fit()
        return self.results

    def diagnostics(self):
        resid = self.results.resid
        lb_table = acorr_ljungbox(resid, lags=[1], return_df=True)
        jb_stat, jb_pvalue, _, _ = jarque_bera(resid)
        return {
            'aic': self.results.aic,
            'bic': self.results.bic,
            'p_ar_l1': self.results.pvalues.get('ar.L1', float('nan')),
            'p_ma_l1': self.results.pvalues.get('ma.L1', float('nan')),
            'lb_pvalue': lb_table['lb_pvalue'].iloc[0],
            'jb_pvalue': jb_pvalue
        }

    def forecast(self, steps=24) -> pd.DataFrame:
        forecast = self.results.get_forecast(steps=steps)
        return forecast.summary_frame()
