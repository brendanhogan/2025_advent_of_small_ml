"""
Async OpenAI client for GPT-4.1 pairwise comparison judging.

Used for GRPO reward signal: given two answers to an MMLU question,
judge which one has better reasoning.
"""

import asyncio
import os
from typing import Optional

import aiohttp


JUDGE_PROMPT_TEMPLATE = """You are a judge evaluating two answers to a multiple choice question.

QUESTION:
{question}

CHOICES:
{choices}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Which answer demonstrates better reasoning? You MUST pick a winner - no ties allowed.
Consider: clarity of explanation, logical steps, correctness of reasoning process.

Reply with ONLY the letter "A" or "B" (nothing else)."""


async def judge_pair(
    session: aiohttp.ClientSession,
    api_key: str,
    question: str,
    choices: str,
    answer_a: str,
    answer_b: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4.1",
) -> str:
    """
    Judge a pair of answers using GPT-4.1.
    
    Returns: "A" or "B" indicating the winner.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        choices=choices,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    
    async with semaphore:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"OpenAI API error: {resp.status} - {error_text}")
                    # Default to A on error (arbitrary but consistent)
                    return "A"
                result = await resp.json()
                response = result["choices"][0]["message"]["content"].strip().upper()
                # Ensure we got a valid response
                if response in ["A", "B"]:
                    return response
                # If model didn't follow instructions, default to A
                return "A"
        except asyncio.TimeoutError:
            print("OpenAI API timeout")
            return "A"
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "A"


async def round_robin_judge(
    question: str,
    choices: str,
    completions: list[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4.1",
    max_concurrent: int = 6,
) -> list[float]:
    """
    Perform round-robin pairwise comparison of all completions.
    
    For N completions, we do N*(N-1)/2 comparisons.
    Each completion's reward is its win rate (wins / total_matches).
    
    Args:
        question: The MMLU question text
        choices: Formatted choices string (A. xxx\nB. yyy\n...)
        completions: List of model completions to compare
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        model: OpenAI model to use for judging
        max_concurrent: Max concurrent API calls
    
    Returns:
        List of rewards (win rates) for each completion, same order as input.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
    
    n = len(completions)
    if n < 2:
        # Can't do pairwise comparison with < 2 completions
        return [0.5] * n
    
    # Generate all pairs (i, j) where i < j
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Track wins for each completion
    wins = [0] * n
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async with aiohttp.ClientSession() as session:
        # Create tasks for all pair comparisons
        async def judge_and_record(i: int, j: int) -> tuple[int, int, str]:
            winner = await judge_pair(
                session=session,
                api_key=api_key,
                question=question,
                choices=choices,
                answer_a=completions[i],
                answer_b=completions[j],
                semaphore=semaphore,
                model=model,
            )
            return i, j, winner
        
        tasks = [judge_and_record(i, j) for i, j in pairs]
        results = await asyncio.gather(*tasks)
        
        # Count wins
        for i, j, winner in results:
            if winner == "A":
                wins[i] += 1
            else:
                wins[j] += 1
    
    # Each completion participates in (n-1) matches
    matches_per_completion = n - 1
    
    # Win rate as reward (0 to 1)
    rewards = [w / matches_per_completion for w in wins]
    
    return rewards


def round_robin_judge_sync(
    question: str,
    choices: str,
    completions: list[str],
    api_key: Optional[str] = None,
    model: str = "gpt-4.1",
    max_concurrent: int = 6,
) -> list[float]:
    """Synchronous wrapper for round_robin_judge."""
    return asyncio.run(round_robin_judge(
        question=question,
        choices=choices,
        completions=completions,
        api_key=api_key,
        model=model,
        max_concurrent=max_concurrent,
    ))


if __name__ == "__main__":
    # Quick test
    import os
    
    question = "What is the capital of France?"
    choices = "A. London\nB. Paris\nC. Berlin\nD. Madrid"
    completions = [
        "<think>France is a country in Europe. Its capital is Paris, which is known for the Eiffel Tower.</think>\nB",
        "<think>Let me think... France... I think it might be London? No wait, London is in England.</think>\nA",
        "<think>Paris is the capital of France. This is a well-known fact.</think>\nB",
        "<think>The capital of France is definitely Paris, a major European city.</think>\nB",
    ]
    
    rewards = round_robin_judge_sync(question, choices, completions)
    print(f"Rewards: {rewards}")

