#!/bin/bash

# Virtual environment name
VENV_NAME="venv"

echo "Starting environment configuration for the Computer Vision project..."

# 1. Create the virtual environment
echo "Creating virtual environment in './$VENV_NAME'..."
python -m venv $VENV_NAME

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully."
    
    # 2. Determine the activation script (Windows vs Linux/macOS)
    if [ -d "$VENV_NAME/Scripts" ]; then
        ACTIVATE_PATH="$VENV_NAME/Scripts/activate"
    else
        ACTIVATE_PATH="$VENV_NAME/bin/activate"
    fi

    # 3. Activate the environment and install dependencies
    echo "Activating the environment..."
    source "$ACTIVATE_PATH"

    echo "Updating pip and installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo "✅ Installation completed successfully."
        echo "To use the environment, run: source $ACTIVATE_PATH"
    else
        echo "❌ Error installing dependencies."
        exit 1
    fi
else
    echo "❌ Error creating the virtual environment. Ensure Python is installed and accessible."
    exit 1
fi
