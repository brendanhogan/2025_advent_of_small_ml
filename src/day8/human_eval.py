#!/usr/bin/env python3
"""
Take the Big Five personality test yourself!
Goes through the test set (60 questions) and computes your personality.
"""

import json
from pathlib import Path

from data import QuestionBank, OCEAN_FULL_NAMES
from scoring import PersonalityScorer


def clear_screen():
    print("\033[2J\033[H", end="")


def print_header():
    print("=" * 60)
    print("        BIG FIVE PERSONALITY TEST (IPIP-NEO)")
    print("=" * 60)
    print()
    print("For each statement, rate how accurately it describes you:")
    print()
    print("  1 = Very Inaccurate")
    print("  2 = Moderately Inaccurate")
    print("  3 = Neither Accurate Nor Inaccurate")
    print("  4 = Moderately Accurate")
    print("  5 = Very Accurate")
    print()
    print("-" * 60)
    print()


def get_answer(question_num: int, total: int, text: str) -> int:
    """Get a valid answer (1-5) from the user."""
    while True:
        print(f"[{question_num}/{total}] \"{text}\"")
        print()
        try:
            response = input("Your answer (1-5): ").strip()
            if response.lower() == 'q':
                raise KeyboardInterrupt
            answer = int(response)
            if 1 <= answer <= 5:
                return answer
            print("Please enter a number between 1 and 5.\n")
        except ValueError:
            print("Please enter a number between 1 and 5.\n")


def print_results(result: dict):
    """Print personality results nicely."""
    print()
    print("=" * 60)
    print("                YOUR PERSONALITY PROFILE")
    print("=" * 60)
    print()
    
    ocean = result["ocean"]
    
    # Sort by score for fun
    sorted_dims = sorted(ocean.items(), key=lambda x: x[1], reverse=True)
    
    for dim, score in sorted_dims:
        full_name = OCEAN_FULL_NAMES[dim]
        bar_len = int((score - 1) * 12)  # Scale 1-5 to 0-48
        bar = "█" * bar_len + "░" * (48 - bar_len)
        
        # Interpret score
        if score < 2.5:
            level = "Low"
        elif score < 3.5:
            level = "Average"
        else:
            level = "High"
        
        print(f"{full_name:20s} {score:.2f}  [{bar}]  {level}")
    
    print()
    print("-" * 60)
    print("FACET BREAKDOWN:")
    print("-" * 60)
    
    for dim in ["N", "E", "O", "A", "C"]:
        full_name = OCEAN_FULL_NAMES[dim]
        print(f"\n{full_name} ({dim}):")
        for facet_name, score in result["facets"][dim].items():
            level = "Low" if score < 2.5 else ("High" if score >= 3.5 else "Avg")
            print(f"  {facet_name:25s} {score:.2f} ({level})")
    
    print()


def main():
    # Load questions
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    
    test_questions = bank.get_test_questions()
    total = len(test_questions)
    
    # Collect answers
    answers = {}
    
    clear_screen()
    print_header()
    print(f"This test has {total} questions. Enter 'q' to quit.\n")
    input("Press Enter to begin...")
    
    try:
        for i, q in enumerate(test_questions, 1):
            print()
            answer = get_answer(i, total, q.text)
            answers[q.id] = answer
            print()
    
    except KeyboardInterrupt:
        print("\n\nTest cancelled.")
        if len(answers) < 10:
            print("Not enough answers to compute results.")
            return
        print(f"Computing partial results from {len(answers)} answers...\n")
    
    # Compute results
    result = scorer.compute_personality(answers)
    
    # Display results
    print_results(result)
    
    # Save to file
    output_path = Path(__file__).parent / "human_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "answers": answers,
            "result": result,
        }, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()

