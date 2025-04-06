#!/usr/bin/env python3
"""
Script to exclude rows from a CSV file based on NCT IDs found in another CSV file.
Creates a new CSV file containing rows from the second CSV that do NOT have NCT IDs matching any in the first CSV.
"""

import csv
import sys
import os

def exclude_nctids(reference_file, source_file, output_file, delimiter=','):
    """
    Create a new CSV with rows from source_file that do NOT have NCT IDs matching any in reference_file.
    
    Args:
        reference_file (str): Path to CSV file containing the reference NCT IDs to exclude
        source_file (str): Path to CSV file from which to filter rows
        output_file (str): Path for the output CSV file with filtered rows
        delimiter (str): Delimiter used in the CSV files (default: ',')
    """
    # Check if files exist
    if not os.path.exists(reference_file):
        print(f"Error: Reference file not found: {reference_file}")
        return False
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return False
    
    # Extract NCT IDs from reference file
    reference_nctids = set()
    with open(reference_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)  # Get the headers
        
        # Find the index of the nctid column
        try:
            nctid_index = headers.index('nctid')
        except ValueError:
            print("Error: 'nctid' column not found in the reference file")
            return False
        
        # Extract NCT IDs
        for row in reader:
            if len(row) > nctid_index and row[nctid_index].strip():
                reference_nctids.add(row[nctid_index].strip())
    
    print(f"Extracted {len(reference_nctids)} unique NCT IDs from reference file {reference_file}")
    
    # Filter rows from source file
    filtered_rows = []
    filtered_count = 0
    kept_count = 0
    
    with open(source_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        source_headers = next(reader)  # Get the headers
        filtered_rows.append(source_headers)  # Add headers to output
        
        # Find the index of the nctid column in source file
        try:
            nctid_index = source_headers.index('nctid')
        except ValueError:
            print("Error: 'nctid' column not found in the source file")
            return False
        
        # Filter rows based on NCT ID
        for row in reader:
            if len(row) > nctid_index:
                nctid = row[nctid_index].strip()
                if nctid and nctid in reference_nctids:
                    # This NCT ID is in the reference file, so exclude this row
                    filtered_count += 1
                else:
                    # NCT ID is not in the reference file, so keep this row
                    filtered_rows.append(row)
                    kept_count += 1
            else:
                # No NCT ID in this row, keep it
                filtered_rows.append(row)
                kept_count += 1
    
    # Write filtered rows to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(filtered_rows)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"NCT ID Exclusion Filter Summary")
    print(f"{'='*50}")
    print(f"\nReference file (NCT IDs to exclude): {reference_file}")
    print(f"Source file: {source_file}")
    print(f"Output file: {output_file}")
    print(f"\nRows excluded: {filtered_count}")
    print(f"Rows kept: {kept_count}")
    print(f"Total rows in output file: {len(filtered_rows)} (including header)")
    
    print(f"\nFiltering complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python exclude_nctids.py <reference.csv> <source.csv> <output.csv> [--tab]")
        print("\nThis script creates a new CSV file with rows from source.csv that do NOT have NCT IDs")
        print("matching any in reference.csv.")
        print("Parameters:")
        print("  reference.csv: File containing the NCT IDs to exclude")
        print("  source.csv: File from which to filter rows")
        print("  output.csv: Path for output file with filtered rows")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    source_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Check if files are tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 4 and sys.argv[4] == '--tab':
        delimiter = '\t'
    else:
        # Try to auto-detect the delimiter by reading the first line
        try:
            with open(reference_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' in first_line and ',' not in first_line:
                    delimiter = '\t'
                    print(f"Auto-detected tab-delimited format")
                elif ',' in first_line:
                    print(f"Using comma as delimiter")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    exclude_nctids(reference_file, source_file, output_file, delimiter)