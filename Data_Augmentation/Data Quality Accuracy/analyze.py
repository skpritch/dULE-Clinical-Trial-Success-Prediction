import matplotlib.pyplot as plt
import numpy as np
import re

# Data parsing
data_lines = [
    "HINT_old: (0.85, 0.07984359711335656, (0.7701564028866434, 0.9298435971133565)), 8046",
    "HINT_new, after improvements: (0.680568769637903, 0.1159749945003803, (0.5645937751375227, 0.7965437641382833)), 5589",
    "ours_only: (0.71428571428, 0.1207363221, (0.59354939218, 0.83502203638)), 6110",
    "all_ours: (0.69742724195, 0.1183556583, (0.56490184281, 0.80797461911)), 11689",
    "total_data: (0.75938, 0.0992, (0.6674, 0.85138)), 19735"
]

# Parse the data
labels = []
accuracies = []
half_widths = []
lower_bounds = []
upper_bounds = []
dataset_sizes = []

for line in data_lines:
    # Split the line at the comma before the dataset size
    main_part, size_part = line.rsplit(',', 1)
    
    # Extract the dataset size
    dataset_size = int(size_part.strip())
    dataset_sizes.append(dataset_size)
    
    # Extract the label
    label_part = main_part.split(':', 1)[0].strip()
    labels.append(label_part)
    
    # Extract accuracy (mean) and confidence interval using regex
    values_part = main_part.split(':', 1)[1].strip()
    # Extract all floating point numbers from the string
    matches = re.findall(r'\d+\.\d+', values_part)
    
    if len(matches) >= 4:
        accuracy = float(matches[0])
        half_width = float(matches[1])
        lower_bound = float(matches[2])
        upper_bound = float(matches[3])
        
        accuracies.append(accuracy)
        half_widths.append(half_width)
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
    else:
        print(f"Warning: Could not parse values from line: {line}")

# Map the labels to more readable names
name_mapping = {
    "HINT_old": "HINT data",
    "HINT_new, after improvements": "New HINT data (improved)",
    "ours_only": "LLM labeled data",
    "all_ours": "All new data",
    "total_data": "Total combined data"
}

readable_labels = [name_mapping.get(label, label) for label in labels]

# Create a color palette (blue theme)
colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(labels)))

# Create the figure
plt.figure(figsize=(10, 7))

# Create scatter plot with error bars
for i, (size, accuracy, half_width, lower, upper, label, color) in enumerate(
    zip(dataset_sizes, accuracies, half_widths, lower_bounds, upper_bounds, readable_labels, colors)):
    
    plt.errorbar(size, accuracy, yerr=half_width, fmt='o', markersize=10, 
                 capsize=8, label=label, color=color, ecolor=color, 
                 elinewidth=2, capthick=2)
    
    # Position the main percentages differently based on the data point
    if label == "Total combined data" or label == "All new data":
        # For the two rightmost points, position to the left of the dot
        plt.text(size - 150, accuracy, 
                f'{accuracy*100:.1f}%', 
                ha='right', va='center', fontsize=12, fontweight='bold')
    elif label == "HINT data":
        # For the third rightmost point, position to the right of the dot
        plt.text(size + 150, accuracy, 
                f'{accuracy*100:.1f}%', 
                ha='left', va='center', fontsize=12, fontweight='bold')
    elif label == "LLM labeled data":
        # Move 71.4% up and to the right
        plt.text(size + 120, accuracy + 0.035, 
                f'{accuracy*100:.1f}%', 
                ha='left', va='center', fontsize=12, fontweight='bold')
    elif label == "New HINT data (improved)":
        # Move 68.1% slightly to the left
        plt.text(size - 120, accuracy, 
                f'{accuracy*100:.1f}%', 
                ha='right', va='center', fontsize=12, fontweight='bold')
    else:
        # For any other points, keep them above the dot
        offset = 0.025  # Small offset to avoid overlap with dot
        plt.text(size, accuracy + offset, 
                f'{accuracy*100:.1f}%', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add upper bound label at the top of the error bar
    plt.text(size, accuracy + half_width + 0.015, 
             f'{upper*100:.1f}%', 
             ha='center', va='bottom', fontsize=11)
    
    # Add lower bound label at the bottom of the error bar
    plt.text(size, accuracy - half_width - 0.015, 
             f'{lower*100:.1f}%', 
             ha='center', va='top', fontsize=11)
             
    # Custom text with positioning based on the dataset
    if "HINT data" == label:
        # For the third point (HINT data), position text further below
        plt.text(size, accuracy - half_width - 0.06,
                 "(C)",
                 ha='center', va='top', fontsize=11, style='italic')
    elif "New HINT data (improved)" == label:
        # For the second point, move text to the right
        plt.text(size + 600, accuracy,
                 "(B) ",
                 ha='left', va='center', fontsize=11, style='italic')
    elif "LLM labeled data" == label:
        # For the first point (leftmost), format with multiple lines and move text left
        plt.text(size-450, accuracy - half_width - 0.085,
                 "(A)",
                 ha='center', va='top', fontsize=11, style='italic')
    elif "All new data" == label:
        # For the fourth point, keep text below
        plt.text(size-120, accuracy - half_width - 0.06,
                 "(D)",
                 ha='center', va='top', fontsize=11, style='italic')
    elif "Total combined data" == label:
        # For the fifth point (new rightmost), add text above
        plt.text(size, accuracy + half_width + 0.08,
                 "(E)",
                 ha='center', va='bottom', fontsize=11, style='italic')

# Add labels and title
plt.xlabel('Training Dataset Size', fontsize=14, fontweight='bold')
plt.ylabel('Labeling Accuracy', fontsize=14, fontweight='bold')
plt.title('Labeling Accuracy vs Training Dataset Size with 66% Confidence Intervals', fontsize=16, fontweight='bold')

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Format y-axis as percentage - CHANGED TO START FROM 0
plt.ylim(0.0, 1.0)
plt.yticks(np.arange(0.0, 1.05, 0.1), [f"{int(x*100)}%" for x in np.arange(0.0, 1.05, 0.1)], fontsize=12)

# Format x-axis with thousand separators
plt.ticklabel_format(axis='x', style='plain')
plt.xticks(fontsize=12)

# Add legend
plt.legend(loc='lower right', frameon=True, fontsize=12)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('accuracy_vs_dataset_size.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print out the actual confidence intervals for verification
print("Scatter plot generated with error bars showing 66% confidence intervals.")
print("Data summary:")
for i, label in enumerate(readable_labels):
    print(f"  {label}: {accuracies[i]:.3f} ± {half_widths[i]:.3f} (CI: {lower_bounds[i]*100:.1f}% - {upper_bounds[i]*100:.1f}%, {dataset_sizes[i]} examples)")