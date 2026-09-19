import numpy as np

from src.forecasting.uncertainty import conformal_quantile


def test_conformal_quantile_uses_upper_empirical_rank():
    residuals = np.arange(1.0, 11.0)
    value = conformal_quantile(residuals, confidence=0.80)

    assert value >= np.quantile(residuals, 0.80)
