from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import List
import os 

import config 
from main import Pipeline

app = FastAPI()
pipe = Pipeline.from_config(config.active_pipeline)


class MarketStateInput(BaseModel):
    date: List[str]   # or datetime strings
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    tic: List


@app.post("/predict")
def predict(data: MarketStateInput):
    d = data.model_dump()                            # turn into regular dict
    df = pd.DataFrame.from_dict(d)                       # list-of-equal-length-arrays → DataFrame
    # now df has columns: 'timestamps', 'open', 'high', etc.
    # next: prepare obs = some feature engineering on df
    global pipe 
    predictions = pipe.predict(df, remove_temp_data=True )

    print("🔍 Predictions type:", type(predictions))
    print("🔍 Predictions:", predictions)
    
    for i in predictions: 
        i['predicted_action'] = i['predicted_action'].tolist()  

    return {"predictionds": predictions}
 
@app.post("/validate")
def validate(data: MarketStateInput):
    d = data.model_dump()                            # turn into regular dict
    df = pd.DataFrame.from_dict(d)    

    global pipe                   # list-of-equal-length-arrays → DataFrame

    # Keep only rows with tickers that are in self.tickers_in_data
    df = df[df["tic"].isin(pipe.tickers_in_data)].copy()

    # Optionally sort by date + ticker to keep it organized
    df.sort_values(by=["date", "tic"], inplace=True)
    df = pipe.actual_data_processing(df)

    os.makedirs("temp", exist_ok=True )
    df_path = os.path.join("temp", "df.csv")
    df.to_csv(df_path, index=False)
    # now df has columns: 'timestamps', 'open', 'high', etc.
    # next: prepare obs = some feature engineering on df
    pipe.validate_saved_models(df_path)
    # Path to results
    results_dir = os.path.join(pipe.checkpoint_dir, "validation_results")

    # Read all CSV files inside validation_results/
    validation_results = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv"):
            fpath = os.path.join(results_dir, fname)
            df_result = pd.read_csv(fpath)
            df = df.fillna(0)
            validation_results.append({
                "filename": fname,
                "data": df_result.to_dict(orient="records")
            })

    print(validation_results)

    return {"validation_results": validation_results}
