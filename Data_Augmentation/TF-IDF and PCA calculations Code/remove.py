#!/usr/bin/env python3
"""
Script to remove all rows with a label of -1 from a CSV file (in-place).
"""

import pandas as pd
import sys
import os
import shutil
from tempfile import NamedTemporaryFile

def filter_csv_inplace(input_file, delimiter=','):
    """
    Remove all rows with a label value of -1 from a CSV file, modifying the file in-place.
    
    Args:
        input_file (str): Path to the CSV file to filter in-place
        delimiter (str): Delimiter used in the CSV file (default: ',')
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        return False
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep=delimiter)
    
    # Make sure 'label' column exists
    if 'label' not in df.columns:
        print(f"Error: Column 'label' not found in the file")
        return False
    
    # Count rows before filtering
    total_rows = len(df)
    
    # Filter out rows where label is -1
    df_filtered = df[df['label'] != -1]
    
    # Count removed rows
    removed_rows = total_rows - len(df_filtered)
    
    if removed_rows == 0:
        print(f"No rows with label -1 found. File remains unchanged.")
        return True
    
    # Create a temporary file to write the filtered data
    with NamedTemporaryFile(mode='w', delete=False) as temp_file:
        # Save the filtered data to the temporary file
        df_filtered.to_csv(temp_file.name, sep=delimiter, index=False)
        temp_file_path = temp_file.name
    
    # Replace the original file with the filtered file
    try:
        shutil.move(temp_file_path, input_file)
        print(f"Successfully updated {input_file} in-place")
    except Exception as e:
        print(f"Error replacing the original file: {e}")
        return False
    
    print(f"\n{'='*50}")
    print(f"CSV In-Place Filtering Summary")
    print(f"{'='*50}")
    print(f"File: {input_file}")
    print(f"Total rows originally: {total_rows}")
    print(f"Rows with label -1 removed: {removed_rows}")
    print(f"Rows remaining in file: {len(df_filtered)}")
    print(f"\nIn-place filtering complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python filter_csv_inplace.py <file.csv> [--tab]")
        print("\nThis script filters a CSV file by removing all rows with a label value of -1.")
        print("The filtering is done in-place, directly modifying the original file.")
        print("\nParameters:")
        print("  file.csv: CSV file to filter in-place")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if file is tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 2 and sys.argv[2] == '--tab':
        delimiter = '\t'
    else:
        # Try to auto-detect the delimiter by reading the first line
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    filter_csv_inplace(input_file, delimiter)