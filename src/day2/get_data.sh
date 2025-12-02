#!/bin/bash

# Download CharXiv data
# This script downloads images from HuggingFace and copies JSON files from the CharXiv repo

# Create data directory
mkdir -p data
cd data

# Download images from HuggingFace if they don't exist
if [ -d "images" ] && [ "$(ls -A images)" ]; then
    echo "Images already exist, skipping download..."
else
    echo "Downloading images from HuggingFace..."
    wget https://huggingface.co/datasets/princeton-nlp/CharXiv/resolve/main/images.zip
    
    # Unzip images
    echo "Extracting images..."
    unzip images.zip
    rm images.zip
fi

# Go back to day1 directory
cd ..

# Get JSON files from CharXiv repo
# Check if CharXiv repo already exists
if [ -d "CharXiv" ]; then
    echo "Using existing CharXiv repo..."
    CHARXIV_DIR="CharXiv"
else
    echo "Cloning CharXiv repo to get JSON files..."
    git clone https://github.com/princeton-nlp/CharXiv.git temp_charxiv
    CHARXIV_DIR="temp_charxiv"
fi

# Copy JSON files to data directory
echo "Copying JSON files..."
cp $CHARXIV_DIR/data/descriptive_val.json data/
cp $CHARXIV_DIR/data/descriptive_test.json data/
cp $CHARXIV_DIR/data/reasoning_val.json data/
cp $CHARXIV_DIR/data/reasoning_test.json data/
cp $CHARXIV_DIR/src/constants.py data/

# Remove temporary repo if we created it
if [ "$CHARXIV_DIR" = "temp_charxiv" ]; then
    echo "Cleaning up..."
    rm -rf temp_charxiv
fi

echo "Done! Data is ready in data/ directory."

# Build the dataset (train/test splits)
echo "Building dataset..."
python build_ds.py

