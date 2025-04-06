import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# Path to your output file
logfile = "../data/mse_out_files/hint_0.9_0.0_2479.out"

# --- 1. Read the .out file ---------------
with open(logfile, "r") as f:
    lines = f.readlines()

# --- 2. Set up regexes for training values ----------------
student_train_re = re.compile(r'loss_student\s*:\s*([\d\.]+)')
teacher_train_re = re.compile(r'loss_teacher\s*:\s*([\d\.]+)')

# For validation losses
student_val_re = re.compile(r'Epoch\s+(\d+):\s+Student Validation Loss\s*=\s*([\d\.]+)')
teacher_val_re = re.compile(r'Epoch\s+(\d+):\s+Teacher Validation Loss\s*=\s*([\d\.]+)')

# --- 3. Parse the file ---------------
# Instead of just values we store (original_index, value)
student_train = []
teacher_train = []

for i, line in enumerate(lines):
    m_stud = student_train_re.search(line)
    m_teach = teacher_train_re.search(line)
    if m_stud:
        val = float(m_stud.group(1))
        # Now keep values in the interval [0.40, 0.80]
        if 0.40 <= val <= 0.80:
            student_train.append((i, val))
    if m_teach:
        val = float(m_teach.group(1))
        if 0.40 <= val <= 0.80:
            teacher_train.append((i, val))

# Parse validation losses (these already include epoch numbers)
student_val = []
teacher_val = []
epoch_nums = []
for line in lines:
    match_stud = student_val_re.search(line)
    if match_stud:
        epoch_nums.append(int(match_stud.group(1)))
        student_val.append(float(match_stud.group(2)))
    match_teach = teacher_val_re.search(line)
    if match_teach:
        teacher_val.append(float(match_teach.group(2)))

# --- 4. Prepare smoothing for training losses ---------------
# Extract the preserved indices and values
if student_train:
    x_student = np.array([t[0] for t in student_train])
    y_student = np.array([t[1] for t in student_train])
else:
    x_student = np.array([])
    y_student = np.array([])

if teacher_train:
    x_teacher = np.array([t[0] for t in teacher_train])
    y_teacher = np.array([t[1] for t in teacher_train])
else:
    x_teacher = np.array([])
    y_teacher = np.array([])

# Smoothing function
def smooth_data(x, y):
    """Returns smooth x and y using a UnivariateSpline if there are enough points."""
    if len(x) > 3:
        spl = UnivariateSpline(x, y, s=0.5 * len(x))
        x_smooth = np.linspace(x[0], x[-1], 500)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    else:
        return x, y

x_student_smooth, student_smooth = smooth_data(x_student, y_student)
x_teacher_smooth, teacher_smooth = smooth_data(x_teacher, y_teacher)

# --- 5. Plotting and saving ---------------

# (a) Combined training plot
plt.figure(figsize=(10,6))
plt.plot(x_student, y_student, 'o', color = 'navy', markersize=3, alpha=0.15, label='Student (raw)')
plt.plot(x_teacher, y_teacher, 'o', color = 'orangered',markersize=3, alpha=0.15, label='Teacher (raw)')
plt.plot(x_student_smooth, student_smooth, '-', linewidth=2, color = 'navy', label='Student (smooth)')
plt.plot(x_teacher_smooth, teacher_smooth, '-', linewidth=2, color = 'orangered', label='Teacher (smooth)')
plt.xlabel("Original index (order of appearance)")
plt.ylabel("Loss")
plt.title("Combined Training Loss")
plt.legend()
plt.savefig("../results/graphs/combined_training_loss.png")
plt.close()

# (b) Student training plot
plt.figure(figsize=(10,6))
plt.plot(x_student, y_student, 'o', color = 'navy', markersize=3, alpha=0.15, label='Student (raw)')
plt.plot(x_student_smooth, student_smooth, '-', linewidth=2, color = 'navy', label='Student (smooth)')
plt.xlabel("Original index (order of appearance)")
plt.ylabel("Student Loss")
plt.title("Student Training Loss")
plt.legend()
plt.savefig("../results/graphs/student_training_loss.png")
plt.close()

# (c) Teacher training plot
plt.figure(figsize=(10,6))
plt.plot(x_teacher, y_teacher, 'o', color = 'orangered', markersize=3, alpha=0.15,label='Teacher (raw)')
plt.plot(x_teacher_smooth, teacher_smooth, '-', linewidth=2, color = 'orangered', label='Teacher (smooth)')
plt.xlabel("Original index (order of appearance)")
plt.ylabel("Teacher Loss")
plt.title("Teacher Training Loss")
plt.legend()
plt.savefig("../results/graphs/teacher_training_loss.png")
plt.close()

# (d) Validation losses plot
plt.figure(figsize=(10,6))
plt.plot(epoch_nums, student_val, 'o-', color = 'navy', label='Student Validation Loss')
plt.plot(epoch_nums, teacher_val, 's-', color = 'orangered', label='Teacher Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Losses")
plt.legend()
plt.savefig("../results/graphs/validation_losses.png")
plt.close()

print("All graphs have been saved to the '../results/graphs' directory.")