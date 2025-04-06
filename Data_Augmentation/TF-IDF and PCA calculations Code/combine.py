#!/usr/bin/env python3
"""
Script to combine three CSV files into one single CSV file.
All input files should have the same column structure.
"""

import csv
import sys
import os

def combine_csv_files(file_paths, output_path, delimiter=','):
    """
    Combine multiple CSV files into a single output file.
    
    Args:
        file_paths (list): List of paths to the CSV files to combine
        output_path (str): Path for the combined output CSV file
        delimiter (str): Delimiter used in the CSV files (default: ',')
    """
    # Check if all files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False
    
    # List to store all rows to write to the output file
    all_rows = []
    headers = None
    
    # Process each input file
    for i, file_path in enumerate(file_paths):
        rows_from_file = 0
        
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=delimiter)
            
            # Get headers from the first file
            if i == 0:
                headers = next(reader)
                all_rows.append(headers)  # Add headers to output rows
            else:
                next(reader)  # Skip headers for subsequent files
            
            # Add all data rows to the combined list
            for row in reader:
                all_rows.append(row)
                rows_from_file += 1
        
        print(f"Added {rows_from_file} rows from {file_path}")
    
    # Write all rows to the output file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)
        writer.writerows(all_rows)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"CSV Combination Summary")
    print(f"{'='*50}")
    print(f"\nInput files:")
    for file_path in file_paths:
        print(f"  - {file_path}")
    print(f"\nOutput file: {output_path}")
    print(f"Total rows in output: {len(all_rows)} (including header)")
    print(f"\nCSV combination complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python combine_csv_files.py <file1.csv> <file2.csv> <file3.csv> <output.csv> [--tab]")
        print("\nThis script combines three CSV files into a single output file.")
        print("All input files should have the same column structure.")
        print("Parameters:")
        print("  file1.csv, file2.csv, file3.csv: Input files to combine")
        print("  output.csv: Path for combined output file")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    file_paths = [sys.argv[1], sys.argv[2], sys.argv[3]]
    output_path = sys.argv[4]
    
    # Check if files are tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 5 and sys.argv[5] == '--tab':
        delimiter = '\t'
    else:
        # Try to auto-detect the delimiter by reading the first line of the first file
        try:
            with open(file_paths[0], 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    combine_csv_files(file_paths, output_path, delimiter)