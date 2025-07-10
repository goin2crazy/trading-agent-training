import numpy as np

import os 
from load_data import load_data
from data_processing import (fill_by_group_interpolation, 
                             add_date_features_and_onehot, 
                             add_MACD_RSI_CC, 
                             PCA_analisys)
from autoencoder_processing import fit_transform_with_autoencoder

def simulate_nan_values(df, nan_fraction=0.05):
    """
    Randomly introduces NaN values into the numeric columns of a DataFrame.
    """
    print(f"\n--- Introducing ~{nan_fraction:.0%} missing values for testing ---")
    df_with_nan = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        # Get a random sample of indices to set to NaN
        nan_indices = df_with_nan.sample(frac=nan_fraction).index
        df_with_nan.loc[nan_indices, col] = np.nan
        
    return df_with_nan

if __name__ == "__main__": 
    data = load_data(start_time="2010-1-1", 
                     end_time='2024-1-1', 
                     stock_names=["CAT"], 
                     )
    
    # 2. Artificially add some NaN values to test the function
    data_with_nans = simulate_nan_values(data, nan_fraction=0.1) # 10% NaNs
    
    # 3. Print the data and NaN counts BEFORE filling
    print("\n--- Data Head (Before Filling) ---")
    print(data_with_nans.head(10))
    print("\n--- Missing Value Counts (Before Filling) ---")
    print(data_with_nans.isnull().sum())
    
    # 4. Call the function to fill the missing values
    print("\n\n--- Running grouped interpolation... ---")
    filled_data = fill_by_group_interpolation(data_with_nans, group_col='tic')
    
    # 5. Print the data and NaN counts AFTER filling
    print("\n--- Data Head (After Filling) ---")
    print(filled_data.head(10))
    print("\n--- Missing Value Counts (After Filling) ---")
    print(filled_data.isnull().sum())

    completed_filled_data = add_MACD_RSI_CC(filled_data)
    print("\n---Added MACD, RSI, CCI values for stocks calculation---")
    print(completed_filled_data.head())

    completed_filled_data = PCA_analisys(completed_filled_data)
    print("\n---Added PCA analized new values---")
    print(completed_filled_data.head())

    dated_data = add_date_features_and_onehot(completed_filled_data)
    print("\n--- Added onehotted day in a week, day in a month and month in year---")
    print(dated_data.head())

    dated_data, trainer = fit_transform_with_autoencoder(dated_data, 
                                                         learning_rate=5e-4, 
                                                         batch_size =8, 
                                                         epochs = 150, 
                                                         latent_space=7,
                                                         checkpoint_dir="autoencoder_checkpoints_7", 
                                                         
                                                         )
    

    compressed_data = dated_data[[col for col in dated_data.columns if col.startswith('enc')]]
    compressed_data.to_csv(os.path.join("datasets", "compressed.csv"))
    print("\n---Data Comperessed and saved into file---")
    print("\n")
    print(compressed_data.head())
    

    
    print("\n--- Test Complete ---")

