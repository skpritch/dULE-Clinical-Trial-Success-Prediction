#!/usr/bin/env python3
"""
Script to split a CSV file into three parts (80/10/10) and append to existing CSV files.
If the target files don't exist, they will be created.
"""

import csv
import sys
import os
import random

def split_and_append_csv(input_file, output_files, split_ratio=[0.8, 0.1, 0.1], delimiter=',', seed=42):
    """
    Split a CSV file into three parts based on the given ratio and append to existing files.
    
    Args:
        input_file (str): Path to the input CSV file
        output_files (list): List of three output file paths to append to
        split_ratio (list): List of ratios for the three splits (should sum to 1)
        delimiter (str): Delimiter used in the CSV file (default: ',')
        seed (int): Random seed for reproducibility
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    # Validate split ratio
    if len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 0.00001:
        print(f"Error: Split ratio must contain 3 values that sum to 1.0")
        return False
    
    # Validate output files
    if len(output_files) != 3:
        print(f"Error: Must provide exactly 3 output file paths")
        return False
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all rows from the input file
    all_rows = []
    header = None
    with open(input_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        header = next(reader)  # Get the header
        all_rows = list(reader)  # Get all data rows
    
    # Shuffle the rows
    random.shuffle(all_rows)
    
    # Calculate split indices
    total_rows = len(all_rows)
    split1_end = int(total_rows * split_ratio[0])
    split2_end = split1_end + int(total_rows * split_ratio[1])
    
    # Create split datasets
    splits = [
        all_rows[:split1_end],          # 80% split
        all_rows[split1_end:split2_end], # 10% split
        all_rows[split2_end:]           # 10% split
    ]
    
    # Append splits to output files
    for i, (split_data, output_file) in enumerate(zip(splits, output_files)):
        # Check if output file exists
        file_exists = os.path.exists(output_file)
        
        # Open file in append mode if it exists, otherwise in write mode
        mode = 'a' if file_exists else 'w'
        with open(output_file, mode, newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=delimiter)
            
            # Write header only if creating a new file
            if not file_exists:
                writer.writerow(header)
            
            # Write data rows
            writer.writerows(split_data)
        
        # Print results
        action = "Appended to" if file_exists else "Created"
        print(f"Split {i+1} ({split_ratio[i]*100:.0f}%): {len(split_data)} rows {action} {output_file}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"CSV Split Summary")
    print(f"{'='*50}")
    print(f"\nInput file: {input_file}")
    print(f"Total rows processed: {total_rows}")
    print(f"\nSplit ratios: {[f'{r*100:.0f}%' for r in split_ratio]}")
    for i, (split_data, output_file) in enumerate(zip(splits, output_files)):
        print(f"Split {i+1} (file {output_file}): {len(split_data)} rows")
    
    print(f"\nSplitting and appending complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python split_append_csv.py <input.csv> <output1.csv> <output2.csv> <output3.csv> [--tab]")
        print("\nThis script splits a CSV file into three parts with an 80/10/10 ratio and appends to existing files.")
        print("Parameters:")
        print("  input.csv: Input CSV file to split")
        print("  output1.csv: File to append the first 80% of rows")
        print("  output2.csv: File to append the second 10% of rows")
        print("  output3.csv: File to append the third 10% of rows")
        print("  --tab: Optional flag for tab-delimited files\n")
        print("Example:")
        print("  python split_append_csv.py data.csv train.csv valid.csv test.csv\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_files = [sys.argv[2], sys.argv[3], sys.argv[4]]
    
    # Check if file is tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 5 and sys.argv[5] == '--tab':
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
    
    split_and_append_csv(input_file, output_files, delimiter=delimiter)