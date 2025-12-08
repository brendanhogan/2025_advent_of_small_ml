"""
Answer extraction from LLM responses.
"""

import re
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[int]:
    """
    Extract answer from \\boxed{N} format.
    
    Returns:
        Integer 1-5 if valid answer found, None otherwise
    """
    # Look for \boxed{N} pattern
    patterns = [
        r'\\boxed\{(\d)\}',      # \boxed{3}
        r'\\boxed\s*\{(\d)\}',   # \boxed {3}
        r'boxed\{(\d)\}',        # boxed{3} (no backslash)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 5:
                return num
    
    return None


def extract_answer_fallback(text: str) -> Optional[int]:
    """
    Fallback extraction if boxed format not found.
    Looks for standalone numbers 1-5 near the end.
    
    Returns:
        Integer 1-5 if found, None otherwise
    """
    # Look for patterns like "my answer is 3" or "I rate this a 4"
    patterns = [
        r'(?:answer|rating|score|choose|select)(?:\s+is)?[\s:]+(\d)',
        r'(?:I\s+)?(?:would\s+)?(?:rate|give|choose|select)(?:\s+(?:this|it|myself))?[\s:]+(\d)',
        r'\b([1-5])\s*$',  # Single digit at end
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            if 1 <= num <= 5:
                return num
    
    return None


def parse_answer(text: str) -> tuple[Optional[int], bool]:
    """
    Parse answer from LLM response.
    
    Returns:
        (answer, used_boxed_format)
        - answer: 1-5 if valid, None if unparseable
        - used_boxed_format: True if found in \boxed{}, False if fallback used
    """
    # Try boxed format first
    answer = extract_boxed_answer(text)
    if answer is not None:
        return answer, True
    
    # Try fallback
    answer = extract_answer_fallback(text)
    if answer is not None:
        return answer, False
    
    return None, False


# Quick test
if __name__ == "__main__":
    test_cases = [
        "I think this describes me well. \\boxed{4}",
        "After reflection, my answer is \\boxed{2}",
        "I would rate this a 3",
        "\\boxed{5} is my rating",
        "This doesn't apply. 1",
        "I'm not sure about this one.",
    ]
    
    for text in test_cases:
        answer, boxed = parse_answer(text)
        print(f"'{text[:50]}...' -> answer={answer}, boxed={boxed}")

