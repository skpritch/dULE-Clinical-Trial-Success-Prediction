#!/usr/bin/env python3
"""
Script to combine two CSV files into a single output file.
Both input files should have the same column structure.
"""

import csv
import sys
import os

def combine_two_csv_files(file1_path, file2_path, output_path, delimiter=','):
    """
    Combine two CSV files into a single output file.
    
    Args:
        file1_path (str): Path to the first CSV file
        file2_path (str): Path to the second CSV file
        output_path (str): Path for the combined output CSV file
        delimiter (str): Delimiter used in the CSV files (default: ',')
    """
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File not found: {file1_path}")
        return False
    
    if not os.path.exists(file2_path):
        print(f"Error: File not found: {file2_path}")
        return False
    
    # List to store all rows to write to the output file
    all_rows = []
    
    # Process the first file
    with open(file1_path, 'r', newline='', encoding='utf-8') as file1:
        reader = csv.reader(file1, delimiter=delimiter)
        headers = next(reader)  # Get the headers
        all_rows.append(headers)  # Add headers to output rows
        
        # Add all rows from the first file
        file1_row_count = 0
        for row in reader:
            all_rows.append(row)
            file1_row_count += 1
        
    print(f"Added {file1_row_count} rows from {file1_path}")
    
    # Process the second file (skip header)
    with open(file2_path, 'r', newline='', encoding='utf-8') as file2:
        reader = csv.reader(file2, delimiter=delimiter)
        next(reader)  # Skip the header
        
        # Add all rows from the second file
        file2_row_count = 0
        for row in reader:
            all_rows.append(row)
            file2_row_count += 1
        
    print(f"Added {file2_row_count} rows from {file2_path}")
    
    # Write all rows to the output file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)
        writer.writerows(all_rows)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"CSV Combination Summary")
    print(f"{'='*50}")
    print(f"\nFile 1: {file1_path} - {file1_row_count} rows")
    print(f"File 2: {file2_path} - {file2_row_count} rows")
    print(f"Output file: {output_path}")
    print(f"Total data rows in output: {file1_row_count + file2_row_count}")
    print(f"Total rows including header: {len(all_rows)}")
    print(f"\nCSV combination complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python combine_two_csvs.py <file1.csv> <file2.csv> <output.csv> [--tab]")
        print("\nThis script combines two CSV files into a single output file.")
        print("Both input files should have the same column structure.")
        print("Parameters:")
        print("  file1.csv, file2.csv: Input files to combine")
        print("  output.csv: Path for combined output file")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Check if files are tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 4 and sys.argv[4] == '--tab':
        delimiter = '\t'
    else:
        # Try to auto-detect the delimiter by reading the first line of file1
        try:
            with open(file1_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    combine_two_csv_files(file1_path, file2_path, output_path, delimiter)