#!/bin/bash

# Check if atom size argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <atom_size>"
    echo "Example: $0 11"
    exit 1
fi

ATOM_SIZE=$1
CHEMBL_DIR="chembl${ATOM_SIZE}"
TREE_DIR="tree${ATOM_SIZE}"
ALL_DIR="${CHEMBL_DIR}/all"
ONLY_TREE_DIR="${CHEMBL_DIR}/only_tree"

echo "========================================="
echo "Processing atom size: ${ATOM_SIZE}"
echo "========================================="

# Step 1: Create 'all' directory and copy files from val and train
echo -e "\n[Step 1] Creating 'all' directory and merging files..."
mkdir -p "${ALL_DIR}"
cp "${CHEMBL_DIR}/val/"*.mol "${ALL_DIR}/" 2>/dev/null || echo "No val files to copy"
cp "${CHEMBL_DIR}/train/"*.mol "${ALL_DIR}/" 2>/dev/null || echo "No train files to copy"
echo "Files merged into ${ALL_DIR}"

# Step 2: Run change_files_name.py on 'all' directory
echo -e "\n[Step 2] Renaming files in 'all' to remove gaps..."
python3 - <<EOF
import os

orig_path = "${ALL_DIR}"

# Get all .mol files and extract their numbers
files = sorted([f for f in os.listdir(orig_path) if f.endswith(".mol")], 
               key=lambda x: int(x.split(".")[0]))
file_numbers = [int(f.split(".")[0]) for f in files]

if file_numbers:
    expected = list(range(1, max(file_numbers) + 1))
    missing_numbers = [num for num in expected if num not in file_numbers]
    #print(f"Detected missing files: {missing_numbers}")
    
    # Rename files to fill gaps
    for num in sorted(file_numbers):
        shift = sum(1 for m in missing_numbers if m < num)
        if shift > 0:
            old_name = os.path.join(orig_path, f"{num}.mol")
            temp_name = os.path.join(orig_path, f"{num - shift}_tmp.mol")
            os.rename(old_name, temp_name)
    
    # Remove _tmp suffix
    for fname in os.listdir(orig_path):
        if fname.endswith('_tmp.mol'):
            temp_path = os.path.join(orig_path, fname)
            final_name = os.path.join(orig_path, fname.replace('_tmp', ''))
            os.rename(temp_path, final_name)
    
    print("Renaming complete!")
EOF

# Step 3: Check for missing files in 'all'
echo -e "\n[Step 3] Checking for missing files in 'all'..."
python3 - <<EOF
import os
import re

folder_path = "${ALL_DIR}"
pattern = re.compile(r'\d+')

file_names = os.listdir(folder_path)
file_numbers = []
for name in file_names:
    match = pattern.search(name)
    if match:
        file_numbers.append(int(match.group()))

if file_numbers:
    file_numbers = sorted(set(file_numbers))
    full_range = set(range(min(file_numbers), max(file_numbers) + 1))
    missing = sorted(full_range - set(file_numbers))
    
    if missing:
        print(f"⚠️ Missing file numbers : check code to print")
    else:
        print("✓ No missing files in 'all' directory")
else:
    print("No files found!")
EOF

# Step 4: Select tree files (no rings)
echo -e "\n[Step 4] Selecting molecules without rings..."
python3 - <<EOF
import shutil
import os

L = []

with open("${CHEMBL_DIR}/smile${ATOM_SIZE}", "r") as f, open("${CHEMBL_DIR}/no_rings.txt", "w") as output_file:
    for index, line in enumerate(f):
        if all(d not in line for d in '12345'):
            L.append(index)
            output_file.write(line)

destination_path = "${ONLY_TREE_DIR}"
orig_path = "${ALL_DIR}"
os.makedirs(destination_path, exist_ok=True)

moved_count = 0
for idx in L:
    file_name = f'{idx}.mol'
    source_file = os.path.join(orig_path, file_name)
    destination_file = os.path.join(destination_path, file_name)
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
        moved_count += 1
    else:
        print(f"File not found: {file_name}")

print(f"Moved {moved_count} files to only_tree")

# Verification
print("\n=== VERIFICATION ===")
with open("${CHEMBL_DIR}/no_rings.txt", "r") as f:
    lines_with_rings = []
    for line_num, line in enumerate(f, 1):
        if any(d in line for d in '123456789'):
            lines_with_rings.append((line_num, line.strip()))
    
    if lines_with_rings:
        print(f"⚠️ WARNING: Found {len(lines_with_rings)} lines with ring digits in no_rings.txt:")
        for line_num, content in lines_with_rings[:5]:
            print(f"  Line {line_num}: {content}")
    else:
        print("✓ Verification passed: no_rings.txt contains no ring digits (1-9)")
EOF

# Step 5: Rename files in 'only_tree' to remove gaps
echo -e "\n[Step 5] Renaming files in 'only_tree' to remove gaps..."
python3 - <<EOF
import os

orig_path = "${ONLY_TREE_DIR}"

files = sorted([f for f in os.listdir(orig_path) if f.endswith(".mol")], 
               key=lambda x: int(x.split(".")[0]))
file_numbers = [int(f.split(".")[0]) for f in files]

if file_numbers:
    expected = list(range(1, max(file_numbers) + 1))
    missing_numbers = [num for num in expected if num not in file_numbers]
    #print(f"Detected missing files in only_tree: {missing_numbers}")
    
    for num in sorted(file_numbers):
        shift = sum(1 for m in missing_numbers if m < num)
        if shift > 0:
            old_name = os.path.join(orig_path, f"{num}.mol")
            temp_name = os.path.join(orig_path, f"{num - shift}_tmp.mol")
            os.rename(old_name, temp_name)
    
    for fname in os.listdir(orig_path):
        if fname.endswith('_tmp.mol'):
            temp_path = os.path.join(orig_path, fname)
            final_name = os.path.join(orig_path, fname.replace('_tmp', ''))
            os.rename(temp_path, final_name)
    
    print("Renaming complete!")
EOF

# Step 6: Check for missing files in 'only_tree'
echo -e "\n[Step 6] Checking for missing files in 'only_tree'..."
python3 - <<EOF
import os
import re

folder_path = "${ONLY_TREE_DIR}"
pattern = re.compile(r'\d+')

file_names = os.listdir(folder_path)
file_numbers = []
for name in file_names:
    match = pattern.search(name)
    if match:
        file_numbers.append(int(match.group()))

if file_numbers:
    file_numbers = sorted(set(file_numbers))
    full_range = set(range(min(file_numbers), max(file_numbers) + 1))
    missing = sorted(full_range - set(file_numbers))
    
    if missing:
        print(f"⚠️ Missing file numbers: {missing}")
    else:
        print("✓ No missing files in 'only_tree' directory")
else:
    print("No files found!")
EOF

# Step 7: Split into train/val
echo -e "\n[Step 7] Splitting into train/val sets..."
python3 - <<EOF
import shutil
import os
import random

random.seed(42)  # For reproducibility

orig_path = "${ONLY_TREE_DIR}"
val_path = "${TREE_DIR}/val"
train_path = "${TREE_DIR}/train"
val_ratio = 0.2

os.makedirs(val_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)

mol_files = [f for f in os.listdir(orig_path) if f.endswith('.mol')]
print(f"Found {len(mol_files)} .mol files in {orig_path}")

val_count = 0
train_count = 0

for file_name in mol_files:
    source_file = os.path.join(orig_path, file_name)
    
    if random.random() < val_ratio:
        dest_file = os.path.join(val_path, file_name)
        shutil.copy2(source_file, dest_file)
        val_count += 1
    else:
        dest_file = os.path.join(train_path, file_name)
        shutil.copy2(source_file, dest_file)
        train_count += 1

print(f"\nSplit complete:")
print(f"  Validation: {val_count} files ({val_count/len(mol_files)*100:.1f}%)")
print(f"  Training: {train_count} files ({train_count/len(mol_files)*100:.1f}%)")
EOF

echo -e "\n========================================="
echo "Pipeline complete for atom size ${ATOM_SIZE}!"
echo "========================================="
