"""
MATH dataset loader and utilities for ES training.
"""

import random
from typing import List, Dict, Any, Tuple
from datasets import load_dataset


class MathDataset:
    """Wrapper for MATH dataset with scoring interface."""
    
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
            import sympy as sp
            
            # Parse both as mathematical expressions
            answer_expr = sp.sympify(answer_norm)
            ground_truth_expr = sp.sympify(ground_truth_norm)
            
            # Check if they're mathematically equivalent
            if sp.simplify(answer_expr - ground_truth_expr) == 0:
                return 1.0
                
        except:
            pass
        
        return 0.0


def load_math_dataset(
    train_size: int = 12000,
    eval_size: int = 20,
    seed: int = 42
) -> Tuple[MathDataset, MathDataset]:
    """
    Load MATH dataset and split into train/eval sets.
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
    """Format a MATH dataset entry into a prompt."""
    return entry['problem']


def extract_math_answer(entry: Dict[str, Any]) -> str:
    """Extract the ground truth answer from a MATH dataset entry."""
    return entry['answer']

