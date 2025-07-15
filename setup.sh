#!/bin/bash

# Brain Tumor Segmentation with Monte Carlo Dropout - Project Setup
# Creates directory structure for uncertainty quantification project

echo "Setting up Brain Tumor Segmentation MC Dropout project..."

# Create main project directory
mkdir -p brain_tumor_mc_dropout
cd brain_tumor_mc_dropout

# Data directories
mkdir -p data/{raw,processed,splits}
mkdir -p data/raw/{images,labels}
mkdir -p data/processed/{train,val,test}

# Source code directories
mkdir -p src/{models,utils,preprocessing,training,inference}
mkdir -p src/uncertainty/{mc_dropout,calibration,visualization}

# Configuration and experiments
mkdir -p configs
mkdir -p experiments/{logs,checkpoints,results}

# Notebooks and analysis
mkdir -p notebooks/{exploratory,analysis,visualization}

# Documentation and outputs
mkdir -p docs
mkdir -p outputs/{predictions,uncertainty_maps,figures}
mkdir -p outputs/statistical_analysis

# Tests
mkdir -p tests/{unit,integration}

# Create initial files
touch README.md
touch requirements.txt
touch .gitignore

# Create initial Python files
touch src/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch src/preprocessing/__init__.py
touch src/training/__init__.py
touch src/inference/__init__.py
touch src/uncertainty/__init__.py
touch src/uncertainty/mc_dropout/__init__.py
touch src/uncertainty/calibration/__init__.py
touch src/uncertainty/visualization/__init__.py

# Create placeholder config files
touch configs/model_config.yaml
touch configs/training_config.yaml
touch configs/data_config.yaml

# Create initial notebooks
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_model_development.ipynb
touch notebooks/03_uncertainty_analysis.ipynb
touch notebooks/04_clinical_evaluation.ipynb

echo "Project structure created successfully!"
echo "Directory tree:"
tree -a -I '__pycache__|*.pyc|.git' || find . -type d | sort