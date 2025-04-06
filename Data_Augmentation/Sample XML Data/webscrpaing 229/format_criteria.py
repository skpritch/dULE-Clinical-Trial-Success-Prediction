import csv
import re

def format_criteria(text):
    """Format inclusion/exclusion criteria to match the target format."""
    
    # Process the text line by line
    lines = text.strip().split('\n')
    result_lines = []
    in_nested_section = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            result_lines.append('')
            continue
        
        # Format section headers
        if re.match(r'^inclusion criteria:?$', stripped, re.IGNORECASE):
            result_lines.append('        Inclusion Criteria:')
            result_lines.append('')
            continue
            
        if re.match(r'^exclusion criteria:?$', stripped, re.IGNORECASE):
            result_lines.append('        Exclusion Criteria:')
            result_lines.append('')
            continue
        
        # Handle bullet points (asterisks or numbers)
        if stripped.startswith('*') or re.match(r'^\d+[\.\)]', stripped):
            # Extract content after bullet
            if stripped.startswith('*'):
                content = re.sub(r'^\*\s*', '', stripped)
            else:
                content = re.sub(r'^\d+[\.\)]\s*', '', stripped)
            
            # Check for nested bullets by examining indentation
            if line.startswith('  *') or line.startswith('  -'):
                result_lines.append(f'             -  {content}')
                in_nested_section = True
            else:
                result_lines.append(f'          -  {content}')
                in_nested_section = False
        else:
            # Regular text or notes
            content = stripped
            
            # Handle Notes and nested content properly
            if content.startswith('Note:'):
                if in_nested_section:
                    result_lines.append(f'             {content}')
                else:
                    result_lines.append(f'          {content}')
            # Handle existing hyphen bullets
            elif content.startswith('-'):
                content = re.sub(r'^-\s*', '', content)
                result_lines.append(f'          -  {content}')
            else:
                # Standard text
                result_lines.append(f'          {content}')
        
        # Convert mathematical symbols - do this for all lines
        result_lines[-1] = re.sub(r'(\s|^)<(\s|\d)', r'\1<=\2', result_lines[-1])
        result_lines[-1] = re.sub(r'(\s|^)>(\s|\d)', r'\1>=\2', result_lines[-1])
    
    return '\n'.join(result_lines)

def process_csv(input_file, output_file):
    """Process the CSV file, formatting the criteria column."""
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        rows = []
        for row in reader:
            if 'criteria' in row:
                row['criteria'] = format_criteria(row['criteria'])
            rows.append(row)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Usage:
input_file = 'updated_merged_outputs.csv'     # Change to your input file
output_file = 'final_data.csv'   # Change to your desired output file

process_csv(input_file, output_file)