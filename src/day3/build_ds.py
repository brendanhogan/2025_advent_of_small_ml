"""
Simple eval harness for CharXiv subset
Prepares 100 images: 75 for training, 25 for testing
"""

import json
import os
import shutil
import random

# Load question mapping from constants
# We need DESCRIPTIVE_GRADING_QMAP to convert question IDs to actual questions
import sys
sys.path.append('data')
from constants import DESCRIPTIVE_GRADING_QMAP

# Set random seed for reproducibility
random.seed(7111994)

# Load data files
print("Loading data files...")
with open('data/descriptive_val.json', 'r') as f:
    descriptive_val = json.load(f)

with open('data/reasoning_val.json', 'r') as f:
    reasoning_val = json.load(f)

# Get all figure IDs that have both descriptive and reasoning questions
all_figure_ids = []
for figure_id_str in descriptive_val.keys():
    if figure_id_str in reasoning_val:
        all_figure_ids.append(int(figure_id_str))

# Select 100 random figure IDs
selected_figure_ids = random.sample(all_figure_ids, 100)

# Split into train (75) and test (25)
train_figure_ids = selected_figure_ids[:75]
test_figure_ids = selected_figure_ids[75:]

# Create directories
os.makedirs('train_images', exist_ok=True)
os.makedirs('test_images', exist_ok=True)
os.makedirs('test_data', exist_ok=True)

# Copy training images (just the images, no questions)
print("Copying training images...")
for figure_id in train_figure_ids:
    src = f'data/{figure_id}.jpg'
    dst = f'train_images/{figure_id}.jpg'
    if os.path.exists(src):
        shutil.copy(src, dst)

# Prepare test data (images + questions + answers)
print("Preparing test data...")
test_data = []

for figure_id in test_figure_ids:
    figure_id_str = str(figure_id)
    
    # Copy image
    src = f'data/{figure_id}.jpg'
    dst = f'test_images/{figure_id}.jpg'
    if os.path.exists(src):
        shutil.copy(src, dst)
    
    # Get descriptive questions
    descriptive_entry = descriptive_val[figure_id_str]
    qids = descriptive_entry['qids']
    answers = descriptive_entry['answers']
    
    descriptive_questions = []
    for i, qid in enumerate(qids):
        question_text = DESCRIPTIVE_GRADING_QMAP[qid]
        answer = answers[i]
        descriptive_questions.append({
            'question': question_text,
            'answer': answer
        })
    
    # Get reasoning question
    reasoning_entry = reasoning_val[figure_id_str]
    reasoning_question = {
        'question': reasoning_entry['query'],
        'answer': reasoning_entry['answer']
    }
    
    # Store everything for this figure
    test_data.append({
        'figure_id': figure_id,
        'image_path': f'test_images/{figure_id}.jpg',
        'descriptive_questions': descriptive_questions,
        'reasoning_question': reasoning_question
    })

# Save test data to JSON
with open('test_data/test_set.json', 'w') as f:
    json.dump(test_data, f, indent=2)

print(f"Done!")
print(f"Training images: {len(train_figure_ids)} in train_images/")
print(f"Test images: {len(test_figure_ids)} in test_images/")
print(f"Test data saved to test_data/test_set.json")
