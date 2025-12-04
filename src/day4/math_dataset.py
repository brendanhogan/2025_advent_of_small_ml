"""
MATH dataset loader and utilities for GRPO training.
"""

import json
import random
from typing import List, Dict, Any, Tuple
from datasets import load_dataset


class MathDataset:
    """Wrapper for MATH dataset with GRPO-compatible interface."""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self._index = 0
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def score_answer(self, answer: str, entry: Dict[str, Any]) -> float:
        """
        Score an extracted answer against the ground truth.
        
        Args:
            answer: The extracted answer from the model
            entry: The dataset entry with ground truth
            
        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        ground_truth = entry['answer']
        
        # Normalize both answers for comparison
        answer_norm = answer.strip().lower()
        ground_truth_norm = ground_truth.strip().lower()
        
        # Remove LaTeX display math delimiters for comparison
        import re
        answer_norm = re.sub(r'\\\(|\\\)', '', answer_norm)
        ground_truth_norm = re.sub(r'\\\(|\\\)', '', ground_truth_norm)
        
        # Direct string match
        if answer_norm == ground_truth_norm:
            return 1.0
        
        # For numerical answers, try to parse and compare
        try:
            # Try to evaluate as mathematical expressions
            import ast
            import sympy as sp
            
            # Parse both as mathematical expressions
            answer_expr = sp.sympify(answer_norm)
            ground_truth_expr = sp.sympify(ground_truth_norm)
            
            # Check if they're mathematically equivalent
            if sp.simplify(answer_expr - ground_truth_expr) == 0:
                return 1.0
                
        except:
            # If parsing fails, fall back to string comparison
            pass
        
        return 0.0


def load_math_dataset(
    train_size: int = 12000,
    eval_size: int = 20,
    seed: int = 42
) -> Tuple[MathDataset, MathDataset]:
    """
    Load MATH dataset and split into train/eval sets.
    
    Args:
        train_size: Number of training examples to use
        eval_size: Number of eval examples to use
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print("Loading MATH dataset...")
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
    
    # Set random seed
    random.seed(seed)
    
    # Get all training data
    train_data = list(dataset['train'])
    
    # Sample eval set from test data
    test_data = list(dataset['test'])
    eval_data = random.sample(test_data, min(eval_size, len(test_data)))
    
    # Sample training data if needed
    if train_size < len(train_data):
        train_data = random.sample(train_data, train_size)
    
    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(eval_data)} eval examples")
    
    return MathDataset(train_data), MathDataset(eval_data)


def format_math_problem(entry: Dict[str, Any]) -> str:
    """
    Format a MATH dataset entry into a prompt.
    
    Args:
        entry: Dataset entry with 'problem' field
        
    Returns:
        Formatted problem string
    """
    return entry['problem']


def extract_math_answer(entry: Dict[str, Any]) -> str:
    """
    Extract the ground truth answer from a MATH dataset entry.
    
    Args:
        entry: Dataset entry with 'answer' field
        
    Returns:
        Ground truth answer string
    """
    return entry['answer']


if __name__ == "__main__":
    # Test the dataset loading
    train_ds, eval_ds = load_math_dataset(train_size=100, eval_size=5)
    
    print("\nSample training examples:")
    for i, example in enumerate(train_ds):
        if i >= 3:
            break
        print(f"\nExample {i+1}:")
        print(f"Problem: {example['problem'][:100]}...")
        print(f"Answer: {example['answer']}")
        print(f"Subject: {example['subject']}")
        print(f"Level: {example['level']}")
    
    print("\nSample eval examples:")
    for i, example in enumerate(eval_ds):
        if i >= 3:
            break
        print(f"\nExample {i+1}:")
        print(f"Problem: {example['problem'][:100]}...")
        print(f"Answer: {example['answer']}")
        print(f"Subject: {example['subject']}")
        print(f"Level: {example['level']}")
    
    # Test answer scoring
    print("\nTesting answer scoring:")
    test_entry = eval_ds[0]
    print(f"Ground truth: {test_entry['answer']}")
    print(f"Correct answer score: {eval_ds.score_answer(test_entry['answer'], test_entry)}")
    print(f"Wrong answer score: {eval_ds.score_answer('wrong', test_entry)}")
