import os
import pandas as pd

def save_trade_data_to_csv(trade_rows, strategy_name, asset_name):
    # Ensure the 'trades_csv' directory exists
    os.makedirs('trades_csv', exist_ok=True)
    
    # Create filename using strategy name and asset name
    filename = f"{strategy_name}_{asset_name}_trades.csv"
    
    # Save the dataframe to a CSV file in the 'trades_csv' directory
    filepath = os.path.join('trades_csv', filename)
    trade_rows.to_csv(filepath)
    print(f"\nTrade data has been saved to {filepath}")

    return filepath

def convert_csv_to_xlsx(csv_filepath):
    # Get the directory and filename from the csv_filepath
    directory, csv_filename = os.path.split(csv_filepath)
    
    # Construct the XLSX filename by replacing the .csv extension with .xlsx
    xlsx_filename = os.path.splitext(csv_filename)[0] + '.xlsx'
    xlsx_filepath = os.path.join(directory, xlsx_filename)
    
    # Check if the CSV file exists
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file {csv_filepath} not found.")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_filepath)
        
        # Write to XLSX file
        df.to_excel(xlsx_filepath, index=False)
        
        print(f"CSV file has been successfully converted to XLSX: {xlsx_filepath}")
        return xlsx_filepath
    except Exception as e:
        print(f"An error occurred while converting the file: {str(e)}")
        return None