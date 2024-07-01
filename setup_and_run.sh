#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Create a virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Step 2: Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Step 3: Install the required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Step 4: Run the data preprocessing script
echo "Running data preprocessing..."
python src/data_preprocessing.py

# Step 5: Train the model
echo "Training the model..."
python src/train.py

# Step 6: Evaluate the model
echo "Evaluating the model..."
python src/evaluate.py

# Step 7: Make predictions
echo "Making predictions..."
python src/predict.py

# Step 8: Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Setup and execution completed successfully."
