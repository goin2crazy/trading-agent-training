from os.path import join, exists # Import 'exists' for checking file existence
from finrl import config_tickers
from finrl.config import DATA_SAVE_DIR
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
import pandas as pd
import os # Import os for creating directories if needed

# Ensure the data save directory exists
if not os.path.exists(DATA_SAVE_DIR):
    os.makedirs(DATA_SAVE_DIR)

def check_file_exist(filepath):
    """
    Checks if a file exists at the given filepath.
    """
    return exists(filepath)

def load_data(start_time, end_time, stock_names=None):
    """
    Loads stock data from local files if available, otherwise downloads it
    using YahooDownloader and saves it locally.

    Args:
        start_time (str): Start date in 'YYYY-MM-DD' format.
        end_time (str): End date in 'YYYY-MM-DD' format.
        stock_names (list, optional): A list of stock tickers. 
                                      Defaults to DOW_30_TICKER if None.

    Returns:
        pd.DataFrame: A DataFrame containing the downloaded or loaded stock data.
    """

    if stock_names is None:
        print("Stock names are not defined, switching to DOW_30_TICKER")
        stock_names = config_tickers.DOW_30_TICKER
    elif type(stock_names) == str: 
        stock_names = [stock_names]

    # List to store individual stock dataframes
    all_stock_dfs = []
    
    # Flag to track if all data for given stocks and period is available locally
    all_data_local = True 

    for stock_name in stock_names:
        filepath = join(DATA_SAVE_DIR, f"{stock_name}-{start_time}-{end_time}.csv") # Added .csv extension

        if check_file_exist(filepath):
            print(f"Loading data for {stock_name} from {filepath}")
            try:
                df_stock = pd.read_csv(filepath)
                all_stock_dfs.append(df_stock)
            except Exception as e:
                print(f"Error loading {stock_name} from file: {e}. Will re-download.")
                all_data_local = False # Set to False if any file fails to load
                break # Break from loop to re-download all if one fails
        else:
            print(f"Data for {stock_name} not found locally. Will download.")
            all_data_local = False
            break # Break from loop to download all if any file is missing

    if all_data_local and all_stock_dfs:
        print("All data found locally and loaded.")
        df = pd.concat(all_stock_dfs, ignore_index=True)
        return df
    else:
        print("Downloading fresh data...")
        df = YahooDownloader(start_date=start_time,
                             end_date=end_time,
                             ticker_list=stock_names).fetch_data()
        
        # Save each stock's data individually after downloading
        for stock_name in stock_names:
            stock_df = df[df['tic'] == stock_name] # Assuming 'tic' is the ticker column
            filepath = join(DATA_SAVE_DIR, f"{stock_name}-{start_time}-{end_time}.csv")
            stock_df.to_csv(filepath, index=False)
            print(f"Saved data for {stock_name} to {filepath}")
        
        return df