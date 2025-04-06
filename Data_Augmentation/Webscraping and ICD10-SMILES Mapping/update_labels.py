import csv

def update_terminated_labels(input_file, output_file):
    """
    Goes through a CSV file, identifies rows where status="terminated",
    changes their label to 0, and writes the result to a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    rows_updated = 0
    
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                # If status is "terminated", change label to 0
                if row['status'].lower() == 'terminated':
                    row['label'] = '0'
                    rows_updated += 1
                    
                writer.writerow(row)
    
    return rows_updated

if __name__ == "__main__":
    input_file = 'merged_output.csv'
    output_file = 'updated_merged_outputs.csv'
    
    rows_updated = update_terminated_labels(input_file, output_file)
    print(f"Processing complete. {rows_updated} rows were updated.")
    print(f"Updated data saved to {output_file}")