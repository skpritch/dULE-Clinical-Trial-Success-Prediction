#!/usr/bin/env python3
"""
ICD-10 Code Format Converter

This script converts ICD-10 codes from storage format (without decimals) 
to display format (with decimals).

Usage:
    python icd10_format_converter.py [input.csv] [output.csv]

If no arguments are provided, it uses codes.csv as input and 
codes_formatted.csv as output.
"""

import csv
import sys
import os

def format_icd10_code(code):
    """
    Convert an ICD-10 code from storage format to display format.
    Example: A066 -> A06.6
    
    Args:
        code: The ICD-10 code in storage format
        
    Returns:
        str: The code in display format (with decimal)
    """
    if len(code) > 3 and '.' not in code:
        return code[:3] + '.' + code[3:]
    return code

def convert_csv_file(input_file, output_file):
    """
    Convert all ICD-10 codes in a CSV file to display format
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    print(f"Converting ICD-10 codes from {input_file} to {output_file}")
    
    # Column index that contains the full ICD-10 code
    code_index = 2  # This is the third column (index 2) in the GitHub dataset
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            # Process each row
            row_count = 0
            converted_count = 0
            
            for row in reader:
                row_count += 1
                
                # Skip if row doesn't have enough columns
                if len(row) <= code_index:
                    writer.writerow(row)  # Write unchanged
                    continue
                
                # Format the code
                if row[code_index]:
                    old_code = row[code_index]
                    new_code = format_icd10_code(old_code)
                    
                    if old_code != new_code:
                        converted_count += 1
                        
                    row[code_index] = new_code
                
                # Write the modified row
                writer.writerow(row)
            
            print(f"Processed {row_count} rows")
            print(f"Converted {converted_count} ICD-10 codes to include decimal points")
            print(f"Conversion complete. Output saved to {output_file}")
    
    except Exception as e:
        print(f"Error converting file: {e}")

def main():
    # Get input and output filenames from command line arguments or use defaults
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    elif len(sys.argv) > 1:
        input_file = sys.argv[1]
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_formatted.csv"
    else:
        input_file = "codes.csv"
        output_file = "codes_formatted.csv"
        print("Using default filenames:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    # Convert the file
    convert_csv_file(input_file, output_file)

if __name__ == "__main__":
    main()