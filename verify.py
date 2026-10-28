"""Verify the standalone model using its bundled historical example."""
from pathlib import Path
import numpy as np
import pandas as pd
from forecast_model import OutageForecast

HERE=Path(__file__).resolve().parent

if __name__=='__main__':
    inputs=pd.read_csv(HERE/'example_input.csv',float_precision='round_trip')
    expected=pd.read_csv(HERE/'example_output.csv',float_precision='round_trip')
    actual=OutageForecast().predict(inputs)
    np.testing.assert_allclose(actual.outage_estimate_percent,expected.outage_estimate_percent,atol=1e-10)
    np.testing.assert_array_equal(actual.risk_level,expected.risk_level)
    np.testing.assert_array_equal(actual.warning,expected.warning)
    assert set(actual.risk_level)=={'Low','Medium','High'}
    assert actual.outage_estimate_percent.between(0,100).all()
    assert (actual.warning.eq(1)==actual.risk_level.eq('High')).all()
    print('Verified: standalone model integrity, historical example replay, all risk labels, and probability range.')
