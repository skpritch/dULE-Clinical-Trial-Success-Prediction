#!/usr/bin/env python3
"""
NCT to ICD-10 Codes Consolidator

This script reads a CSV file with NCT IDs and ICD-10 codes, 
then creates a new CSV file with one row per NCT ID, consolidating all ICD-10 codes.

Usage:
    python nct_icd10_consolidator.py input.csv output.csv

If no arguments are provided, it uses icd10_mapping.csv as input and 
nct_consolidated.csv as output.
"""

import csv
import sys
import os
import re
from collections import defaultdict

def extract_codes(code_text):
    """
    Extract ICD-10 codes from formatted text.
    Examples: 
    - "L28.1 (Prurigo Nodularis)" -> ["L28.1"]
    - "D61.01 (aplastic)" -> ["D61.01"]
    
    Args:
        code_text: String containing ICD-10 codes with descriptions
        
    Returns:
        list: List of ICD-10 codes
    """
    # If the text is empty, return empty list
    if not code_text:
        return []
    
    # This regex pattern matches codes like L28.1, D61.01, etc.
    pattern = r'([A-Z]\d+\.\d+)'
    
    # Find all matches
    codes = re.findall(pattern, code_text)
    return codes

def consolidate_icd10_codes(input_file, output_file):
    """
    Consolidate ICD-10 codes by NCT ID
    
    Args:
        input_file: Path to input CSV file (NCT ID, Disease, ICD-10 Codes)
        output_file: Path to output CSV file (NCT ID, ICD-10 Codes)
    """
    print(f"Consolidating ICD-10 codes from {input_file} to {output_file}")
    
    # Dictionary to store ICD-10 codes by NCT ID
    nct_to_codes = defaultdict(set)
    
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader, None)  # Skip header row
            
            # Process each row
            row_count = 0
            code_count = 0
            
            for row in reader:
                row_count += 1
                
                if len(row) >= 3:
                    nct_id = row[0]
                    icd10_text = row[2]
                    
                    # Extract codes from text
                    codes = extract_codes(icd10_text)
                    code_count += len(codes)
                    
                    # Add to set for this NCT ID
                    for code in codes:
                        nct_to_codes[nct_id].add(code)
        
        # Write output file
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["NCT ID", "ICD-10 Codes"])
            
            # Write one row per NCT ID
            for nct_id, codes in sorted(nct_to_codes.items()):
                # Format the codes list
                if codes:
                    # Sort codes for consistent output
                    sorted_codes = sorted(list(codes))
                    # Format as string list with single quotes
                    codes_formatted = "['{}']".format("', '".join(sorted_codes))
                else:
                    codes_formatted = "[]"
                
                writer.writerow([nct_id, codes_formatted])
            
        print(f"Processed {row_count} rows with {code_count} ICD-10 codes")
        print(f"Created {len(nct_to_codes)} consolidated entries")
        print(f"Consolidation complete. Output saved to {output_file}")
    
    except Exception as e:
        print(f"Error consolidating file: {e}")

def main():
    # Get input and output filenames from command line arguments or use defaults
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) > 1:
        input_file = sys.argv[1]
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_consolidated.csv"
    else:
        input_file = "results/icd10_mapping.csv"
        output_file = "results/nct_consolidated.csv"
        print("Using default filenames:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Consolidate the file
    consolidate_icd10_codes(input_file, output_file)

if __name__ == "__main__":
    main()