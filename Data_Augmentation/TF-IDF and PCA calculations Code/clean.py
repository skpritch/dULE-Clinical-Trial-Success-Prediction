import pandas as pd
import os
import shutil
from datetime import datetime

def trim_csv(file_path, output_path=None, encoding='utf-8', create_backup=True):
    """
    Trims a CSV file by removing all columns that come after the 'criteria' column.
    
    Args:
        file_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the output CSV file. If None,
                                    the input file will be overwritten.
        encoding (str, optional): Encoding of the input file. Default is 'utf-8'.
        create_backup (bool, optional): Whether to create a backup of the original file.
                                       Default is True.
    """
    if output_path is None:
        output_path = file_path
    
    # Ensure the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Create a backup if requested
    if create_backup and output_path == file_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.{timestamp}.bak"
        try:
            shutil.copy2(file_path, backup_path)
            print(f"Backup created at '{backup_path}'")
        except Exception as e:
            print(f"Warning: Failed to create backup: {e}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        print(f"Error reading the file: {e}")
        print("Trying with different encodings...")
        
        # Try with different encodings
        for enc in ['latin1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(file_path, encoding=enc)
                print(f"Successfully read the file with encoding '{enc}'.")
                encoding = enc
                break
            except Exception:
                continue
        else:
            print("Failed to read the file with common encodings.")
            return
    
    # Get the column names
    columns = df.columns.tolist()
    
    # Find the index of the "criteria" column (case-insensitive)
    criteria_col = None
    for col in columns:
        if col.lower() == "criteria":
            criteria_col = col
            break
    
    if criteria_col is None:
        print("Error: 'criteria' column not found in the CSV file.")
        return
    
    criteria_index = columns.index(criteria_col)
    
    # Keep only columns up to and including the "criteria" column
    columns_to_keep = columns[:criteria_index + 1]
    
    # Create a new DataFrame with only the columns we want to keep
    trimmed_df = df[columns_to_keep]
    
    # Save the modified DataFrame back to the specified output file
    try:
        trimmed_df.to_csv(output_path, index=False, encoding=encoding)
        print(f"CSV file has been modified and saved to '{output_path}'.")
        print(f"All columns after '{criteria_col}' have been removed.")
        print(f"Remaining columns: {', '.join(columns_to_keep)}")
    except Exception as e:
        print(f"Error writing to the output file: {e}")

# Usage example
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "valid_tfidf.csv"  
    
    # If you want to create a new file instead of modifying the original:
    # output_path = "trimmed_data.csv"
    # trim_csv(file_path, output_path=output_path)
    
    # This will create a backup and modify the original file
    trim_csv(file_path, create_backup=True)