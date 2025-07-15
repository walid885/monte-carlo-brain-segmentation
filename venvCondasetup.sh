#!/bin/bash
# Usage: . ./script_name.sh (tab)

# Deactivate conda base environment if active
if [[ "$CONDA_DEFAULT_ENV" == "base" ]] || [[ -n "$CONDA_DEFAULT_ENV" ]]; then
    echo "Deactivating conda environment: $CONDA_DEFAULT_ENV"
    conda deactivate
fi

# Check if we're in the correct directory
if [[ ! -d "MC_BT" ]]; then
    echo "MC_BT virtual environment not found in current directory."
    echo "Current directory: $(pwd)"
    echo "Please run this script from ~/Desktop/monte-carlo-brain-segmentation/"
    exit 1
fi

echo "Activating MC_BT virtual environment..."
source MC_BT/bin/activate

if [[ "$VIRTUAL_ENV" == *"MC_BT"* ]]; then
    echo "MC_BT environment activated successfully."
    echo "Python path: $(which python)"
    echo "Virtual environment: $VIRTUAL_ENV"
    # Force remove conda from prompt
    unset CONDA_DEFAULT_ENV
    unset CONDA_PREFIX
else
    echo "Failed to activate MC_BT environment."
    echo "Please ensure MC_BT/bin/activate exists in the current directory."
    exit 1
fi