import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from stockstats import StockDataFrame

def fill_by_group_interpolation(df: pd.DataFrame, 
                                group_col: str, 
                                time_col: str = "date", 
                                method: str = 'linear', 
                                limit_direction: str = 'forward', 
                                order: int = None) -> pd.DataFrame:
    # Create a copy to avoid modifying the original DataFrame
    df_filled = df.copy()

    # Identify numeric columns that need filling (excluding the group and time columns if they are numeric)
    numeric_cols = df_filled.select_dtypes(include=np.number).columns.tolist()
    if group_col in numeric_cols:
        numeric_cols.remove(group_col)
    if time_col and time_col in numeric_cols:
        numeric_cols.remove(time_col)

    # Sort the DataFrame by the group column and then by the time column (if provided)
    # This is crucial for correct time-series interpolation within each group.
    if time_col:
        # Ensure the time_col is in datetime format if it's not already, for proper sorting
        if pd.api.types.is_string_dtype(df_filled[time_col]):
            df_filled[time_col] = pd.to_datetime(df_filled[time_col], errors='coerce')
        df_filled = df_filled.sort_values(by=[group_col, time_col]).reset_index(drop=True)
    else:
        # If no time_col, just sort by group_col to ensure contiguous groups
        df_filled = df_filled.sort_values(by=[group_col]).reset_index(drop=True)


    # Apply interpolation to each numeric column within each group
    # The transform() method ensures the output is aligned with the original DataFrame's index
    # after the grouped operation.
    for col in numeric_cols:
        df_filled[col] = df_filled.groupby(group_col)[col].transform(
            lambda x: x.interpolate(
                method=method,
                limit_direction=limit_direction,
                order=order
            )
        )
        # After group-wise interpolation, use ffill() and bfill() to catch any remaining
        # NaNs that might occur at the very beginning or end of a group if
        # limit_direction didn't cover them (e.g., a group starting with NaN and limit_direction='backward').
        df_filled[col] = df_filled[col].ffill().bfill()

    return df_filled

# Add some more metrics 
# Umm, okay this feature is already realized in FinRL 
# def add_MACD_RSI_CC(df) -> pd.DataFrame: 
#     assert "tic" in df.columns, "Please add the ticket name column into dataframe, it needed for sorting"
#     assert "date" in df.columns, "Please add the date column into dataframe, it needed for sorting"

#     required_columns = ['close', 'open', 'high', 'low', 'volume']
#     for col in required_columns: 
#         assert col in df.columns, f"Please add the {col} column into dataframe it needed for calculation"


#     # StockDataFrame requires specific column names: 'close', 'open', 'high', 'low', 'volume'
#     # and it works better if each stock is handled separately
#     df = df.sort_values(by=['tic', 'date'])  # make sure data is sorted
#     stock_df = StockDataFrame.retype(df.copy())
#     stock_df['rsi']
#     # Now your DataFrame has new columns
#     print("Calculated the new columns, Update: ", stock_df.columns)
#     return stock_df

# Add the day in week, day in month and month in year
# to give a better understaning of time
# One-hot encoded
def add_date_features_and_onehot(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid modifying the original DataFrame directly
    df_processed = df.copy()

    # Check if 'date' is in the index and reset if so
    if df_processed.index.name == 'date':
        print("Info: 'date' found in DataFrame index. Resetting index to make it a column.")
        df_processed.reset_index(inplace=True)
    
    # Assert that 'date' column exists after potential index reset
    if "date" not in df_processed.columns:
        raise ValueError("Required 'date' column not found in DataFrame or its index.")

    # Ensure the 'date' column is in datetime format
    # errors='coerce' will turn unparseable dates into NaT (Not a Time)
    df_processed['date'] = pd.to_datetime(df_processed['date'], errors='coerce')

    # Drop rows where date conversion failed (if any)
    if df_processed['date'].isnull().any():
        print("Warning: Some dates could not be parsed and corresponding rows were dropped.")
        df_processed.dropna(subset=['date'], inplace=True)
        
    if df_processed.empty:
        print("Error: DataFrame is empty after date parsing. Cannot add features.")
        return df_processed

    # Extract new temporal features
    # dayofweek: Monday=0, Sunday=6
    df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
    df_processed['day_of_week'] = df_processed['day_of_week'].apply(lambda i: str(i))
    # day: Day of the month (1-31)
    df_processed['day_of_month'] = df_processed['date'].dt.day
    df_processed['day_of_month'] = df_processed['day_of_month'].apply(lambda i: str(i))
    # month: Month of the year (1-12)
    df_processed['month_of_year'] = df_processed['date'].dt.month
    df_processed['month_of_year'] = df_processed['month_of_year'].apply(lambda i: str(i))

    # Define the columns to be one-hot encoded
    features_to_onehot = ['day_of_week', 'day_of_month', 'month_of_year']

    # Apply one-hot encoding
    # prefix ensures the new column names are descriptive (e.g., 'day_of_week_0', 'month_of_year_1')
    # drop_first=True avoids multicollinearity (for linear models) by dropping the first category
    df_onehot_encoded = pd.get_dummies(
        df_processed[features_to_onehot],
        prefix=features_to_onehot,
        drop_first=False # Set to False to keep all categories, or True if you want to drop the first
    )
    
    # Convert boolean True/False to 1/0 integers
    df_onehot_encoded = df_onehot_encoded.astype(int)

    # Concatenate the one-hot encoded features back to the original DataFrame
    # Drop the original categorical columns if you only want the one-hot encoded versions
    # If you want to keep both, remove the 'drop' part
    df_final = pd.concat([df_processed.drop(columns=features_to_onehot), df_onehot_encoded], axis=1)

    return df_final

# Principal Component Analysis of data for better conseptual understanding 
def PCA_analisys(df: pd.DataFrame, n_components = 3) -> pd.DataFrame: 
    """IMPORTANT: Use before the one-hot encoding the date and time"""
    df = df.copy() 

    features = df.select_dtypes(include=np.number).columns.tolist()
    features_df = df[features]

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    # Keep, say, 3 principal components:
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_features)

    for i in range(n_components): 
        df[f'pca{i}'] = principal_components[:, i]

    print("Explained variance ratios:", pca.explained_variance_ratio_)
    print("e.g. [0.45, 0.25, 0.15] means those 3 comps explain 85% of the variance")


    return df 
