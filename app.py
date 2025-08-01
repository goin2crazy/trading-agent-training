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


@app.post("/predict")
def predict(data: MarketStateInput):
    d = data.model_dump()                            # turn into regular dict
    df = pd.DataFrame(d)                       # list-of-equal-length-arrays → DataFrame
    # now df has columns: 'timestamps', 'open', 'high', etc.
    # next: prepare obs = some feature engineering on df
    global pipe 
    predictions = pipe.predict(df, remove_temp_data=True )

    return {"predictionds": predictions}
 
@app.post("/validate")
def predict(data: MarketStateInput):
    d = data.model_dump()                            # turn into regular dict
    df = pd.DataFrame(d)                       # list-of-equal-length-arrays → DataFrame

    os.makedirs("temp", exist_ok=True )
    df_path = os.path.join("temp", "df.csv")
    df.to_csv(df_path)
    # now df has columns: 'timestamps', 'open', 'high', etc.
    # next: prepare obs = some feature engineering on df
    global pipe 
    pipe.validate_saved_models(df_path)
    # Path to results
    results_dir = os.path.join(pipe.checkpoint_dir, "validation_results")

    # Read all CSV files inside validation_results/
    validation_results = []
    for fname in os.listdir(results_dir):
        if fname.endswith(".csv"):
            fpath = os.path.join(results_dir, fname)
            df_result = pd.read_csv(fpath)
            validation_results.append({
                "filename": fname,
                "data": df_result.to_dict(orient="records")
            })

    return {"validation_results": validation_results}
