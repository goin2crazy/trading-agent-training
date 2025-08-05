from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

# Pick one ticker for test
ticker = DOW_30_TICKER  # e.g., 'AAPL'

# Load data for a short period
df = YahooDownloader(start_date="2023-04-01",
                     end_date="2024-07-15",
                     ticker_list=ticker).fetch_data()

print(df.head())

test_payload = {
    "date": df["date"].astype(str).tolist(),
    "open": df["open"].tolist(),
    "high": df["high"].tolist(),
    "low": df["low"].tolist(),
    "close": df["close"].tolist(),
    "volume": df["volume"].tolist(),
    "tic": df["tic"].tolist()   
}

import requests
import json

# Send to /predict
# resp_predict = requests.post("http://127.0.0.1:8000/predict", json=test_payload)
# print("PREDICT:", resp_predict.status_code)
# print(json.dumps(resp_predict.json(), indent=2))

# Send to /validate
resp_validate = requests.post("http://127.0.0.1:8000/validate", json=test_payload)
print("VALIDATE:", resp_validate.status_code)
print(json.dumps(resp_validate.json(), indent=2))
