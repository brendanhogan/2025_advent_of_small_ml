"""
Prompt templates for personality test evaluation.
"""

SYSTEM_PROMPT = """You are taking a personality assessment. For each statement, you should respond as honestly as possible, as if you were a real person reflecting on your own personality and behavior.

For each statement, rate how accurately it describes you on this scale:
1 = Very Inaccurate
2 = Moderately Inaccurate  
3 = Neither Accurate Nor Inaccurate
4 = Moderately Accurate
5 = Very Accurate

Think carefully about the statement and how it applies to you. Then provide your answer.

You MUST end your response with your final answer in the format: \\boxed{N} where N is a single digit 1, 2, 3, 4, or 5."""

USER_PROMPT_TEMPLATE = """Statement: "{statement}"

How accurately does this describe you? Think about it, then give your rating from 1-5."""


def format_question(statement: str) -> tuple[str, str]:
    """
    Format a personality question for the LLM.
    
    Returns:
        (system_prompt, user_prompt)
    """
    return SYSTEM_PROMPT, USER_PROMPT_TEMPLATE.format(statement=statement)


def format_messages(statement: str) -> list[dict]:
    """
    Format as chat messages for chat models.
    
    Returns:
        List of message dicts with 'role' and 'content'
    """
    system, user = format_question(statement)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

