#!/usr/bin/env python3
"""
Script to remove rows from the second CSV file where the NCT ID already exists in the first CSV file.
This ensures no duplicated NCT IDs when combining datasets.
"""

import csv
import sys
import os

def remove_overlapping_nctids(file1_path, file2_path, output_path, delimiter=','):
    """
    Remove rows from the second CSV file where the NCT ID already exists in the first CSV file.
    
    Args:
        file1_path (str): Path to the reference CSV file (NCT IDs to check against)
        file2_path (str): Path to the CSV file to filter (remove overlapping NCT IDs)
        output_path (str): Path for the filtered output CSV file
        delimiter (str): Delimiter used in the CSV files (default: ',')
    """
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File not found: {file1_path}")
        return False
    
    if not os.path.exists(file2_path):
        print(f"Error: File not found: {file2_path}")
        return False
    
    # Extract all NCT IDs from the first file
    nctids_file1 = set()
    with open(file1_path, 'r', newline='', encoding='utf-8') as file1:
        reader = csv.reader(file1, delimiter=delimiter)
        headers = next(reader)  # Get the headers
        
        # Find the index of the nctid column
        try:
            nctid_index = headers.index('nctid')
        except ValueError:
            print("Error: 'nctid' column not found in the first CSV file")
            return False
        
        # Extract NCT IDs
        for row in reader:
            if len(row) > nctid_index and row[nctid_index].strip():
                nctids_file1.add(row[nctid_index].strip())
    
    print(f"Extracted {len(nctids_file1)} unique NCT IDs from {file1_path}")
    
    # Process the second file to exclude rows with NCT IDs already in file1
    rows_to_keep = []
    removed_count = 0
    
    with open(file2_path, 'r', newline='', encoding='utf-8') as file2:
        reader = csv.reader(file2, delimiter=delimiter)
        file2_headers = next(reader)  # Get the headers
        rows_to_keep.append(file2_headers)  # Keep the headers
        
        # Find the index of the nctid column in file2
        try:
            nctid_index = file2_headers.index('nctid')
        except ValueError:
            print("Error: 'nctid' column not found in the second CSV file")
            return False
        
        # Filter rows based on NCT ID
        for row in reader:
            if len(row) > nctid_index:
                nctid = row[nctid_index].strip()
                if nctid and nctid in nctids_file1:
                    # This NCT ID is in file1, so skip this row
                    removed_count += 1
                else:
                    # NCT ID is not in file1, so keep this row
                    rows_to_keep.append(row)
            else:
                # No NCT ID in this row, keep it
                rows_to_keep.append(row)
    
    # Write the filtered data to the output file
    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile, delimiter=delimiter)
        writer.writerows(rows_to_keep)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Overlap Removal Summary")
    print(f"{'='*50}")
    print(f"\nFile 1 (reference): {file1_path}")
    print(f"File 2 (to filter): {file2_path}")
    print(f"Output file: {output_path}")
    print(f"\nRows removed (overlapping NCT IDs): {removed_count}")
    print(f"Rows kept in output file: {len(rows_to_keep) - 1} data rows + 1 header row")
    print(f"\nOverlap removal complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python remove_overlapping_nctids.py <file1.csv> <file2.csv> <output.csv> [--tab]")
        print("\nThis script removes rows from file2 where the NCT ID already exists in file1.")
        print("Parameters:")
        print("  file1.csv: Reference file (NCT IDs to check against)")
        print("  file2.csv: File to filter (remove overlapping NCT IDs)")
        print("  output.csv: Path for filtered output file")
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
    
    remove_overlapping_nctids(file1_path, file2_path, output_path, delimiter)