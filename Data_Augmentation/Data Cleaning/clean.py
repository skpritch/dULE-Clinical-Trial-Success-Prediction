#!/usr/bin/env python3
"""
Script to restructure the clinical trials CSV:
1. Remove all TF-IDF columns and preprocessed_criteria
2. Replace criteria column content with PCA vector
"""

import pandas as pd
import sys
import os

def restructure_csv(input_file, output_file, delimiter=','):
    """
    Restructure a CSV file by removing TF-IDF columns and replacing criteria with PCA vector.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the restructured CSV file
        delimiter (str): Delimiter used in the CSV file (default: ',')
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file, sep=delimiter)
    
    # Get column names and identify columns to keep
    all_columns = df.columns.tolist()
    
    # Define columns to keep - those up to and including criteria
    criteria_index = all_columns.index('criteria')
    pca_vector_index = all_columns.index('pca_vector')
    
    # Keep columns before criteria, plus pca_vector
    keep_columns = all_columns[:criteria_index + 1] 
    
    # Create a new DataFrame with the columns to keep
    print(f"Restructuring data: removing TF-IDF columns and replacing criteria with PCA vector...")
    new_df = df[keep_columns].copy()
    
    # Replace criteria content with pca_vector content
    new_df['criteria'] = df['pca_vector']
    
    # Save the restructured data
    print(f"Saving restructured data to {output_file}...")
    new_df.to_csv(output_file, sep=delimiter, index=False)
    
    print(f"\n{'='*50}")
    print(f"CSV Restructuring Summary")
    print(f"{'='*50}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Original columns: {len(all_columns)}")
    print(f"Kept columns: {len(keep_columns)}")
    print(f"Removed columns: {len(all_columns) - len(keep_columns)}")
    print(f"Criteria column now contains PCA vector data")
    print(f"\nRestructuring complete!")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python restructure_csv.py <input.csv> <output.csv> [--tab]")
        print("\nThis script restructures a clinical trials CSV file by:")
        print("  1. Removing all TF-IDF columns and preprocessed_criteria")
        print("  2. Replacing criteria column content with PCA vector data")
        print("\nParameters:")
        print("  input.csv: Input CSV file to restructure")
        print("  output.csv: Output CSV file to save restructured data")
        print("  --tab: Optional flag for tab-delimited files\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if file is tab-delimited or comma-delimited
    delimiter = ','
    if len(sys.argv) > 3 and sys.argv[3] == '--tab':
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
    
    restructure_csv(input_file, output_file, delimiter)