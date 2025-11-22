#!/bin/bash

# This script runs multiple training experiments for LAURACV.py, iterating through
# a predefined list of model sizes and data directories.

# --- Configuration ---
# IMPORTANT: You must replace these placeholder paths with the correct paths to your datasets.
DATA_MSD_PATH="/path/to/your/msd_dataset"
DATA_NSCLC_PATH="/path/to/your/nsclc_dataset"

# Define the lists of parameters to iterate over
MODEL_SIZES=("small" "tiny" "mobile")
DATA_DIRECTORIES=("$DATA_MSD_PATH" "$DATA_NSCLC_PATH")
DATA_NAMES=("MSD" "NSCLC") # Friendly names for logging

# --- Main Execution Loop ---

echo "🚀 Launching batch of training experiments..."
echo "================================================="

# Check if the Python script exists before starting the loop
if [ ! -f "LAURACV.py" ]; then
    echo "❌ Error: LAURACV.py not found in the current directory. Aborting."
    exit 1
fi

TOTAL_RUNS=$((${#MODEL_SIZES[@]} * ${#DATA_DIRECTORIES[@]}))
CURRENT_RUN=1

# Loop over each model size
for model in "${MODEL_SIZES[@]}"; do
    # Loop over each data directory
    for i in ${!DATA_DIRECTORIES[@]}; do
        data_dir=${DATA_DIRECTORIES[$i]}
        data_name=${DATA_NAMES[$i]}

        echo ""
        echo "--- [Starting Run ${CURRENT_RUN} of ${TOTAL_RUNS}] ---"
        echo "   - 📝 Model: '${model}'"
        echo "   - 📁 Dataset: '${data_name}'"
        echo "   - 🗂️  Path: '${data_dir}'"
        echo "-------------------------------------"
        
        # Check if the data directory path is still a placeholder
        if [[ "${data_dir}" == "/path/to/your/"* ]]; then
            echo "⚠️  Warning: The data path '${data_dir}' appears to be a placeholder."
            echo "   Please edit this script and set the correct path before running."
            echo "   Skipping this run."
            ((CURRENT_RUN++))
            continue
        fi

        # Execute the python script with the current combination of parameters
        stdbuf -oL python LAURACV.py --MODEL_SIZE "${model}" --DATA_DIR "${data_dir}"

        # Check the exit code of the python script
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Training failed for Model '${model}' with Dataset '${data_name}'."
            echo "   Aborting remaining runs."
            exit 1
        fi

        # Rename log files to prevent them from being overwritten in the next run
        echo "   - 🏷️ Renaming log files..."
        shopt -s nullglob
        for logfile in lightweight_auravit*.log; do
            new_name="${model}_${data_name}_${logfile}"
            mv "$logfile" "$new_name"
            echo "     - Renamed '$logfile' to '$new_name'"
        done
        shopt -u nullglob # Unset nullglob to restore default behavior

        echo "✅ [Finished Run ${CURRENT_RUN} of ${TOTAL_RUNS}]"
        ((CURRENT_RUN++))
    done
done

echo ""
echo "================================================="
echo "🎉 All experiments completed successfully!"
echo ""

# --- Git Archiving ---
echo "📦 Archiving log files to GitHub..."

# Add all generated log files. The pattern matches files like 'small_MSD_lightweight_auravit_training.log'.
git add ./*_training.log

# Check if there are any staged log files before committing
if [ -n "$(git status --porcelain -- ./*_training.log)" ]; then
    git commit -m "feat: Add training logs from all experiments"
    echo "   - ✅ Log files committed."
    
    echo "   - 🚀 Pushing to remote repository..."
    git push
    
    if [ $? -ne 0 ]; then
        echo "   - ❌ ERROR: Failed to push to GitHub."
        exit 1
    else
        echo "   - ✅ Log files successfully pushed to GitHub."
    fi
else
    echo "   - ⚠️ No new log files were generated to commit."
fi

echo ""
echo "✨ All done."
