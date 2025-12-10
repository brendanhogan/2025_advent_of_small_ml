"""
Extract prompts and their eval scores from GEPA run_log.json
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_log", type=Path, default=Path("gepa_qwen7b_run/run_log.json"))
    parser.add_argument("--output", type=Path, default=Path("gepa_qwen7b_run/prompt_history.txt"))
    args = parser.parse_args()
    
    with open(args.run_log) as f:
        data = json.load(f)
    
    # Collect all eval steps with their prompts and scores
    eval_results = []
    
    for step_str, step_data in data.get("steps", {}).items():
        if "eval" in step_data:
            eval_info = step_data["eval"]
            metrics = eval_info.get("metrics", {})
            prompt = eval_info.get("best_prompt", "")
            
            pass_at_1 = metrics.get("pass_at_1", 0)
            avg_format = metrics.get("avg_format_reward", 0)
            num_candidates = metrics.get("num_candidates", 0)
            
            eval_results.append({
                "step": int(step_str),
                "pass_at_1": pass_at_1,
                "avg_format_reward": avg_format,
                "num_candidates": num_candidates,
                "prompt": prompt,
            })
    
    # Sort by step
    eval_results.sort(key=lambda x: x["step"])
    
    # Find the best one
    best = max(eval_results, key=lambda x: x["pass_at_1"])
    
    # Write output
    with open(args.output, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("GEPA PROMPT EVOLUTION - SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Best prompt first
        f.write("*" * 80 + "\n")
        f.write("★★★ BEST PROMPT (Step {}, Pass@1: {:.2f}%) ★★★\n".format(
            best["step"], best["pass_at_1"]
        ))
        f.write("*" * 80 + "\n\n")
        f.write(best["prompt"] + "\n\n")
        f.write("*" * 80 + "\n\n\n")
        
        # All steps
        f.write("=" * 80 + "\n")
        f.write("FULL HISTORY\n")
        f.write("=" * 80 + "\n\n")
        
        for result in eval_results:
            is_best = result["step"] == best["step"]
            marker = " ★ BEST" if is_best else ""
            
            f.write("-" * 80 + "\n")
            f.write(f"Step {result['step']}{marker}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Pass@1: {result['pass_at_1']:.2f}%\n")
            f.write(f"Avg Format Reward: {result['avg_format_reward']:.4f}\n")
            f.write(f"Num Candidates: {result['num_candidates']}\n")
            f.write(f"\nPrompt:\n")
            f.write("-" * 40 + "\n")
            f.write(result["prompt"] + "\n")
            f.write("-" * 40 + "\n\n\n")
    
    print(f"Saved to {args.output}")
    print(f"\nBest result: Step {best['step']} with Pass@1 = {best['pass_at_1']:.2f}%")


if __name__ == "__main__":
    main()

