#!/usr/bin/env python3
"""
Script to extract rows from a second CSV file based on NCT IDs found in a first CSV file.
Creates a new CSV containing only rows from the second CSV where the NCT ID exists in the first CSV.
"""

import csv
import sys
import os

def extract_rows_by_nctids(reference_file, source_file, output_file, delimiter=','):
    """
    Extract rows from source_file where the NCT ID exists in reference_file.
    
    Args:
        reference_file (str): Path to CSV file containing the reference NCT IDs
        source_file (str): Path to CSV file from which to extract rows
        output_file (str): Path for the output CSV file with extracted rows
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
    
    # Extract matching rows from source file
    matching_rows = []
    with open(source_file, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=delimiter)
        source_headers = next(reader)  # Get the headers
        matching_rows.append(source_headers)  # Add headers to output
        
        # Find the index of the nctid column in source file
        try:
            nctid_index = source_headers.index('nctid')
        except ValueError:
            print("Error: 'nctid' column not found in the source file")
            return False
        
        # Extract rows where NCT ID is in the reference set
        matches_found = 0
        for row in reader:
            if len(row) > nctid_index and row[nctid_index].strip() in reference_nctids:
                matching_rows.append(row)
                matches_found += 1
    
    # Write matching rows to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=delimiter)
        writer.writerows(matching_rows)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"NCT ID Extraction Summary")
    print(f"{'='*50}")
    print(f"\nReference file: {reference_file}")
    print(f"Source file: {source_file}")
    print(f"Output file: {output_file}")
    print(f"\nNCT IDs in reference file: {len(reference_nctids)}")
    print(f"Matching rows found in source file: {matches_found}")
    print(f"Total rows in output file: {len(matching_rows)} (including header)")
    
    # Check for reference NCT IDs not found in source
    match_percentage = (matches_found / len(reference_nctids)) * 100 if reference_nctids else 0
    print(f"\nMatch rate: {match_percentage:.1f}% of reference NCT IDs found in source file")
    
    print(f"\nExtraction complete!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_by_nctids.py <reference.csv> <source.csv> <output.csv> [--tab]")
        print("\nThis script extracts rows from source.csv where the NCT ID exists in reference.csv.")
        print("Parameters:")
        print("  reference.csv: File containing the reference NCT IDs")
        print("  source.csv: File from which to extract rows")
        print("  output.csv: Path for output file with extracted rows")
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
    
    extract_rows_by_nctids(reference_file, source_file, output_file, delimiter)