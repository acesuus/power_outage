"""Score prepared day-ahead feature inputs without training or outcome labels."""
import argparse
from pathlib import Path
import pandas as pd
from forecast_model import OutageForecast

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    destination=Path(args.output)
    if destination.exists():raise FileExistsError('Choose a new output filename.')
    result=OutageForecast().predict(pd.read_csv(args.input,float_precision='round_trip'))
    result.to_csv(destination,index=False)
    print(result[['date','recloser','outage_estimate_percent','risk_level']].to_string(index=False))
