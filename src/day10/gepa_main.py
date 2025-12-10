"""
GEPA - Genetic-Pareto Prompt Optimizer for MATH dataset.

A clean, readable implementation of GEPA that optimizes system prompts
using reflective evolution. This parallels the GRPO training script
for fair comparison.

Key ideas:
  1. Maintain a pool of candidate prompts, each scored on validation instances
  2. Select candidates using Pareto-based sampling (diversity-aware)
  3. Evaluate on a minibatch, collect traces (inputs, outputs, feedback)
  4. Use an LLM to reflect on failures and propose improved prompts
  5. Accept improved prompts, evaluate on full validation set, update Pareto front

Unlike GRPO which updates model weights, GEPA evolves the prompt text itself.
Uses vLLM for fast batched generation (same as GRPO).
"""

import os
import json
import random
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from tqdm import tqdm
import textwrap

import torch
from openai import OpenAI

# Reuse the same modules as GRPO
import utils
import llms
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class GEPALogger:
    """Pretty logging for GEPA optimization process."""
    
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    def __init__(self, output_dir: str, verbose: bool = True):
        self.output_dir = output_dir
        self.verbose = verbose
        self.log_file = os.path.join(output_dir, "optimization_log.txt")
        
        # Create fresh log file
        with open(self.log_file, "w") as f:
            f.write(f"GEPA Optimization Log - Started {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
    
    def _write(self, text: str, also_print: bool = True):
        """Write to log file and optionally print."""
        # Strip ANSI codes for file
        import re
        clean_text = re.sub(r'\033\[[0-9;]*m', '', text)
        with open(self.log_file, "a") as f:
            f.write(clean_text + "\n")
        if also_print and self.verbose:
            print(text)
    
    def section(self, title: str):
        """Print a major section header."""
        self._write(f"\n{self.BOLD}{'='*70}{self.ENDC}")
        self._write(f"{self.BOLD}{self.CYAN}  {title}{self.ENDC}")
        self._write(f"{self.BOLD}{'='*70}{self.ENDC}\n")
    
    def subsection(self, title: str):
        """Print a subsection header."""
        self._write(f"\n{self.YELLOW}--- {title} ---{self.ENDC}")
    
    def step_header(self, step: int, total_steps: int, num_candidates: int):
        """Print step header."""
        self._write(f"\n{self.BOLD}{self.HEADER}{'─'*70}{self.ENDC}")
        self._write(f"{self.BOLD}{self.HEADER}  STEP {step}/{total_steps}  |  Candidates: {num_candidates}{self.ENDC}")
        self._write(f"{self.BOLD}{self.HEADER}{'─'*70}{self.ENDC}")
    
    def parent_selected(self, parent_idx: int, prompt: str, avg_score: float, method: str):
        """Log parent selection."""
        self._write(f"\n{self.BLUE}📌 Parent Selected:{self.ENDC} Candidate #{parent_idx} (avg score: {avg_score:.3f}) via {method}")
        self._write(f"{self.DIM}   Prompt preview: {prompt[:100]}...{self.ENDC}")
    
    def minibatch_eval(self, scores: list, traces: list):
        """Log minibatch evaluation results."""
        correct = sum(1 for s in scores if s >= 1.0)
        total = len(scores)
        self._write(f"\n{self.BLUE}📊 Minibatch Evaluation:{self.ENDC} {correct}/{total} correct")
        
        for i, (score, trace) in enumerate(zip(scores, traces)):
            status = f"{self.GREEN}✓{self.ENDC}" if score >= 1.0 else f"{self.RED}✗{self.ENDC}"
            fmt_status = "" if trace.format_ok else f" {self.YELLOW}(format error){self.ENDC}"
            self._write(f"   {status} Example {i+1}: extracted='{trace.extracted_answer or 'None'}' target='{trace.target_answer}'{fmt_status}")
    
    def reflective_dataset(self, dataset: str):
        """Log the reflective dataset being sent to optimizer."""
        self._write(f"\n{self.BLUE}📝 Reflective Dataset (sent to optimizer LLM):{self.ENDC}")
        # Show first 1500 chars
        preview = dataset[:2000]
        if len(dataset) > 2000:
            preview += f"\n... [{len(dataset) - 2000} more characters]"
        for line in preview.split('\n'):
            self._write(f"   {self.DIM}{line}{self.ENDC}")
    
    def optimizer_call(self, model: str, prompt: str):
        """Log the call to optimizer LLM."""
        self._write(f"\n{self.BLUE}🤖 Calling Optimizer LLM:{self.ENDC} {model}")
        self._write(f"{self.DIM}   Meta-prompt length: {len(prompt)} chars{self.ENDC}")
        
        # Save full meta prompt to separate file for inspection
        meta_prompt_file = os.path.join(self.output_dir, "last_meta_prompt.txt")
        with open(meta_prompt_file, "w") as f:
            f.write(prompt)
        self._write(f"{self.DIM}   Full meta-prompt saved to: last_meta_prompt.txt{self.ENDC}")
    
    def proposed_prompt(self, old_prompt: str, new_prompt: str):
        """Log the proposed new prompt."""
        self._write(f"\n{self.CYAN}💡 PROPOSED NEW PROMPT:{self.ENDC}")
        self._write(f"{self.BOLD}{'─'*50}{self.ENDC}")
        
        # Show the full new prompt (formatted nicely)
        for line in new_prompt.split('\n'):
            self._write(f"   {line}")
        
        self._write(f"{self.BOLD}{'─'*50}{self.ENDC}")
        
        # Show diff info
        if old_prompt.strip() == new_prompt.strip():
            self._write(f"   {self.YELLOW}⚠ No change from parent prompt{self.ENDC}")
        else:
            old_len = len(old_prompt)
            new_len = len(new_prompt)
            self._write(f"   {self.DIM}Length: {old_len} → {new_len} ({new_len - old_len:+d} chars){self.ENDC}")
        
        # Save new prompt to file
        proposed_file = os.path.join(self.output_dir, "last_proposed_prompt.txt")
        with open(proposed_file, "w") as f:
            f.write(new_prompt)
    
    def new_prompt_eval(self, old_scores: list, new_scores: list):
        """Log evaluation of new prompt on same minibatch."""
        old_sum = sum(old_scores)
        new_sum = sum(new_scores)
        total = len(old_scores)
        
        if new_sum > old_sum:
            self._write(f"\n{self.GREEN}📈 New Prompt Evaluation: {new_sum}/{total} vs {old_sum}/{total} (IMPROVED!){self.ENDC}")
        elif new_sum < old_sum:
            self._write(f"\n{self.RED}📉 New Prompt Evaluation: {new_sum}/{total} vs {old_sum}/{total} (worse){self.ENDC}")
        else:
            self._write(f"\n{self.YELLOW}📊 New Prompt Evaluation: {new_sum}/{total} vs {old_sum}/{total} (same){self.ENDC}")
    
    def acceptance_decision(self, accepted: bool, old_score: float, new_score: float, val_score: float = None):
        """Log acceptance decision."""
        if accepted:
            self._write(f"\n{self.GREEN}{'='*50}{self.ENDC}")
            self._write(f"{self.GREEN}✅ PROMPT ACCEPTED!{self.ENDC}")
            self._write(f"{self.GREEN}   Minibatch: {old_score} → {new_score}{self.ENDC}")
            if val_score is not None:
                self._write(f"{self.GREEN}   Validation avg: {val_score:.3f}{self.ENDC}")
            self._write(f"{self.GREEN}{'='*50}{self.ENDC}")
        else:
            self._write(f"\n{self.RED}❌ Prompt rejected (no improvement: {old_score} → {new_score}){self.ENDC}")
    
    def eval_results(self, step: int, pass_at_k: float, k: int, avg_format: float, best_idx: int):
        """Log evaluation results."""
        self._write(f"\n{self.GREEN}{'*'*50}{self.ENDC}")
        self._write(f"{self.GREEN}📊 EVALUATION @ Step {step}{self.ENDC}")
        self._write(f"{self.GREEN}   Pass@{k}: {pass_at_k:.2f}%{self.ENDC}")
        self._write(f"{self.GREEN}   Avg Format Reward: {avg_format:.3f}{self.ENDC}")
        self._write(f"{self.GREEN}   Best Candidate: #{best_idx}{self.ENDC}")
        self._write(f"{self.GREEN}{'*'*50}{self.ENDC}")
    
    def candidate_status(self, candidates: list):
        """Log current candidate pool status."""
        self._write(f"\n{self.BLUE}📋 Candidate Pool Status:{self.ENDC}")
        for i, c in enumerate(candidates):
            avg = c.avg_score()
            num_evals = len(c.val_scores)
            parent = f"parent=#{c.parent_idx}" if c.parent_idx is not None else "seed"
            self._write(f"   #{i}: avg={avg:.3f} ({num_evals} evals) [{parent}] step={c.created_at_step}")
    
    def info(self, msg: str):
        """Log info message."""
        self._write(f"{self.DIM}{msg}{self.ENDC}")
    
    def success(self, msg: str):
        """Log success message."""
        self._write(f"{self.GREEN}{msg}{self.ENDC}")
    
    def warning(self, msg: str):
        """Log warning message."""
        self._write(f"{self.YELLOW}⚠ {msg}{self.ENDC}")
    
    def error(self, msg: str):
        """Log error message."""
        self._write(f"{self.RED}❌ {msg}{self.ENDC}")


# ============================================================================
# GEPA DATA STRUCTURES
# ============================================================================

@dataclass
class Candidate:
    """A candidate prompt with its evaluation history."""
    prompt: str
    parent_idx: Optional[int] = None  # Index of parent candidate (None for seed)
    
    # Per-instance scores on validation set: {instance_idx: score}
    val_scores: dict = field(default_factory=dict)
    
    # When this candidate was created
    created_at_step: int = 0
    
    def avg_score(self) -> float:
        """Average score across all evaluated instances."""
        if not self.val_scores:
            return 0.0
        return sum(self.val_scores.values()) / len(self.val_scores)


@dataclass  
class ReflectionTrace:
    """Trace of a single evaluation for reflection."""
    question: str
    target_answer: str
    model_response: str
    extracted_answer: str
    is_correct: bool
    format_ok: bool


# ============================================================================
# GENERATION - Uses vLLM's OpenAI-compatible API for fast batched generation
# ============================================================================

def generate_completions_vllm(
    vllm_openai_client: OpenAI,
    model_name: str,
    prompt_text: str,
    num_completions: int,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Generate multiple completions for a single prompt using vLLM.
    """
    response = vllm_openai_client.completions.create(
        model=model_name,
        prompt=prompt_text,
        n=num_completions,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    
    completions = [choice.text for choice in response.choices]
    return completions


def generate_batch_vllm(
    vllm_openai_client: OpenAI,
    model_name: str,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Generate one completion for each of multiple prompts in a single batched call.
    This is much faster than calling generate_completions_vllm in a loop.
    """
    response = vllm_openai_client.completions.create(
        model=model_name,
        prompt=prompts,  # List of prompts - generates 1 completion each
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
    )
    
    # Responses come back in order
    completions = [choice.text for choice in response.choices]
    return completions


def generate_batch_local(
    model,
    tokenizer,
    prompts: list[str],
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Generate one completion for each prompt using local model with batching.
    """
    # Tokenize all prompts with padding
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=True
    ).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract completions (new tokens only)
    completions = []
    for i, output in enumerate(outputs):
        prompt_len = inputs["attention_mask"][i].sum().item()
        new_tokens = output[prompt_len:]
        completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    
    return completions


def generate_completions_local(
    model,
    tokenizer,
    prompt_text: str,
    num_completions: int,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    """
    Generate multiple completions using local model with batched generation.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].repeat(num_completions, 1).to(model.device)
    attention_mask = inputs["attention_mask"].repeat(num_completions, 1).to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Extract only the new tokens (completions)
    prompt_len = input_ids.size(1)
    completion_ids = outputs[:, prompt_len:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return completions


# ============================================================================
# CORE GEPA FUNCTIONS
# ============================================================================

def evaluate_prompt_on_batch(
    system_prompt: str,
    batch: list,
    dataset,
    tokenizer,
    model,
    model_name: str,
    vllm_client,
    use_vllm: bool,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> tuple[list[float], list[ReflectionTrace]]:
    """
    Evaluate a prompt on a batch of problems (one completion per problem).
    
    Returns:
        scores: List of scores (1.0 for correct, 0.0 for incorrect)
        traces: List of ReflectionTrace objects for building reflective dataset
    """
    scores = []
    traces = []
    
    for entry in batch:
        question = format_math_problem(entry)
        target_answer = extract_math_answer(entry)
        
        # Build the full prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Generate one completion
        if use_vllm:
            completions = generate_completions_vllm(
                vllm_client, model_name, prompt_text,
                num_completions=1, temperature=temperature, max_tokens=max_tokens
            )
        else:
            completions = generate_completions_local(
                model, tokenizer, prompt_text,
                num_completions=1, temperature=temperature, max_tokens=max_tokens
            )
        
        model_response = completions[0]
        
        # Score the response
        extracted = utils.extract_answer(model_response)
        format_reward = utils.check_format(model_response)
        format_ok = format_reward > 0
        
        if not format_ok:
            is_correct = False
        elif extracted:
            is_correct = dataset.score_answer(answer=extracted, entry=entry) == 1.0
        else:
            is_correct = False
        
        score = 1.0 if is_correct else 0.0
        scores.append(score)
        
        traces.append(ReflectionTrace(
            question=question,
            target_answer=target_answer,
            model_response=model_response,
            extracted_answer=extracted,
            is_correct=is_correct,
            format_ok=format_ok,
        ))
    
    return scores, traces


def build_reflective_dataset(traces: list[ReflectionTrace]) -> str:
    """
    Format traces into a reflective dataset for the optimizer LLM.
    """
    examples = []
    
    for i, trace in enumerate(traces):
        status = "✓ CORRECT" if trace.is_correct else "✗ INCORRECT"
        format_status = "Format OK" if trace.format_ok else "FORMAT ERROR (missing tags)"
        
        example = f"""
## Example {i+1}: {status}

### Input Question
{trace.question}

### Model Response
{trace.model_response[:1500]}{'...[truncated]' if len(trace.model_response) > 1500 else ''}

### Evaluation
- Format: {format_status}
- Extracted Answer: {trace.extracted_answer or '(none)'}
- Correct Answer: {trace.target_answer}
"""
        examples.append(example)
    
    return "\n".join(examples)


def propose_new_prompt(
    current_prompt: str,
    reflective_dataset: str,
    openai_client: OpenAI,
    optimizer_model: str,
    logger: GEPALogger = None,
) -> tuple[str, str]:
    """
    Use an LLM to reflect on the evaluation traces and propose an improved prompt.
    
    Returns:
        (new_prompt, meta_prompt) - the proposed prompt and the full meta-prompt sent to optimizer
    """
    
    meta_prompt = f"""You are an expert prompt engineer. I have a system prompt for a math-solving AI assistant, and I've collected examples of its successes and failures.

## Current System Prompt
```
{current_prompt}
```

## Evaluation Results (showing what the AI produced with this prompt)
{reflective_dataset}

## Your Task

Analyze the failures above and propose an IMPROVED system prompt. Focus on:

1. **Format Issues**: If the AI failed to use the required <think></think> and <answer></answer> tags, add clearer instructions about this.

2. **Reasoning Errors**: If the AI made mathematical mistakes, consider adding:
   - Reminders to check work
   - Specific strategies for common problem types
   - Instructions to be careful about specific error patterns you observe

3. **Answer Extraction**: If correct answers weren't extracted properly, clarify the expected format.

4. **Domain-Specific Knowledge**: If you notice patterns in what problems fail, add targeted guidance.

Keep the prompt concise but comprehensive. The prompt should be self-contained - the AI won't see these examples, only your improved prompt.

Provide ONLY the new system prompt, wrapped in triple backticks:
```
your improved prompt here
```"""

    if logger:
        logger.optimizer_call(optimizer_model, meta_prompt)

    try:
        response = openai_client.chat.completions.create(
            model=optimizer_model,
            messages=[{"role": "user", "content": meta_prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        output = response.choices[0].message.content.strip()
        
        # Extract prompt from backticks
        import re
        match = re.search(r'```(?:\w*\n)?(.*?)```', output, re.DOTALL)
        if match:
            new_prompt = match.group(1).strip()
        else:
            new_prompt = output.strip()
        
        if logger:
            logger.proposed_prompt(current_prompt, new_prompt)
        
        return new_prompt, meta_prompt
            
    except Exception as e:
        if logger:
            logger.error(f"Optimizer LLM error: {e}")
        else:
            print(f"Optimizer LLM error: {e}")
        return current_prompt, meta_prompt


def select_candidate_pareto(candidates: list[Candidate], rng: random.Random) -> int:
    """
    Select a candidate using Pareto-based sampling.
    """
    if len(candidates) == 1:
        return 0
    
    all_val_ids = set()
    for c in candidates:
        all_val_ids.update(c.val_scores.keys())
    
    if not all_val_ids:
        return 0
    
    # For each validation instance, find which candidates achieve the best score
    best_per_instance: dict[int, set[int]] = {}
    
    for val_id in all_val_ids:
        best_score = -float('inf')
        best_candidates = set()
        
        for cand_idx, cand in enumerate(candidates):
            score = cand.val_scores.get(val_id, -float('inf'))
            if score > best_score:
                best_score = score
                best_candidates = {cand_idx}
            elif score == best_score:
                best_candidates.add(cand_idx)
        
        best_per_instance[val_id] = best_candidates
    
    # Get unique candidates that are best on at least one instance
    pareto_candidates = set()
    for best_set in best_per_instance.values():
        pareto_candidates.update(best_set)
    
    # Remove dominated candidates
    def is_dominated(cand_idx: int, other_idx: int) -> bool:
        dominated_all = True
        strictly_worse_on_one = False
        
        for val_id in all_val_ids:
            score_cand = candidates[cand_idx].val_scores.get(val_id, 0)
            score_other = candidates[other_idx].val_scores.get(val_id, 0)
            
            if score_cand > score_other:
                dominated_all = False
                break
            if score_other > score_cand:
                strictly_worse_on_one = True
        
        return dominated_all and strictly_worse_on_one
    
    non_dominated = set(pareto_candidates)
    for cand_idx in list(pareto_candidates):
        for other_idx in pareto_candidates:
            if cand_idx != other_idx and is_dominated(cand_idx, other_idx):
                non_dominated.discard(cand_idx)
                break
    
    if not non_dominated:
        non_dominated = pareto_candidates if pareto_candidates else set(range(len(candidates)))
    
    # Count frequency and sample
    frequency = {idx: 0 for idx in non_dominated}
    for val_id, best_set in best_per_instance.items():
        for cand_idx in best_set:
            if cand_idx in frequency:
                frequency[cand_idx] += 1
    
    sampling_list = []
    for cand_idx, freq in frequency.items():
        sampling_list.extend([cand_idx] * max(freq, 1))
    
    return rng.choice(sampling_list)


def select_candidate_best(candidates: list[Candidate]) -> int:
    """Simple selection: just pick the best average score."""
    best_idx = 0
    best_score = candidates[0].avg_score()
    
    for i, cand in enumerate(candidates[1:], 1):
        score = cand.avg_score()
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx


# ============================================================================
# EVALUATION - Uses batched generation like GRPO (FAST!)
# ============================================================================

def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric."""
    if n - c < k:
        return 1.0
    prob_all_wrong = 1.0
    for i in range(k):
        prob_all_wrong *= (n - c - i) / (n - i)
    return 1.0 - prob_all_wrong


def run_evaluation(
    system_prompt: str,
    eval_ds,
    tokenizer,
    model,
    model_name: str,
    vllm_client,
    use_vllm: bool,
    num_completions: int,
    pass_at_k: int,
    temperature: float,
    max_tokens: int,
) -> tuple[float, float, list[dict]]:
    """
    Run evaluation with pass@k metric.
    Batches ALL prompts together for maximum speed.
    For 20 problems × 20 completions = 400 prompts in one vLLM call.
    """
    eval_data = list(eval_ds)
    
    # Build all prompts - repeat each prompt num_completions times
    all_prompts = []
    problem_info = []  # (entry, question, target_answer, problem_type)
    
    for entry in eval_data:
        question = format_math_problem(entry)
        target_answer = extract_math_answer(entry)
        problem_type = f"{entry['subject']}_level_{entry['level']}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Repeat this prompt num_completions times
        for _ in range(num_completions):
            all_prompts.append(prompt_text)
        
        problem_info.append((entry, question, target_answer, problem_type))
    
    # Generate ALL completions in one batched call
    if use_vllm:
        all_completions = generate_batch_vllm(
            vllm_client, model_name, all_prompts,
            temperature=temperature, max_tokens=max_tokens
        )
    else:
        all_completions = generate_batch_local(
            model, tokenizer, all_prompts,
            temperature=temperature, max_tokens=max_tokens
        )
    
    # Process results - group completions by problem
    pass_at_k_scores = []
    format_total = 0.0
    examples = []
    
    for prob_idx, (entry, question, target_answer, problem_type) in enumerate(problem_info):
        # Extract this problem's completions
        start_idx = prob_idx * num_completions
        end_idx = start_idx + num_completions
        completions = all_completions[start_idx:end_idx]
        
        # Score all completions
        extracted_answers = [utils.extract_answer(t) for t in completions]
        format_rewards = [utils.check_format(t) for t in completions]
        
        correctness = []
        for ea, f in zip(extracted_answers, format_rewards):
            if f < 0:
                correctness.append(0.0)
            elif ea:
                correctness.append(float(eval_ds.score_answer(answer=ea, entry=entry) == 1.0))
            else:
                correctness.append(0.0)
        
        # Compute pass@k
        num_correct = sum(correctness)
        pak = compute_pass_at_k(num_completions, int(num_correct), pass_at_k)
        pass_at_k_scores.append(pak)
        
        # Format reward
        avg_format = sum(format_rewards) / len(format_rewards)
        format_total += avg_format
        
        examples.append({
            "question": question,
            "target_answer": target_answer,
            "problem_type": problem_type,
            "completions": [
                {
                    "text": t[:500] + "..." if len(t) > 500 else t,
                    "extracted_answer": ea,
                    "correct": int(c),
                    "format_reward": float(f),
                }
                for t, ea, c, f in zip(completions, extracted_answers, correctness, format_rewards)
            ],
            "num_correct": int(num_correct),
            "pass_at_k": pak,
            "avg_format_reward": avg_format,
        })
    
    avg_pass_at_k = (sum(pass_at_k_scores) / len(pass_at_k_scores)) * 100
    avg_format = format_total / len(eval_data)
    
    return avg_pass_at_k, avg_format, examples


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GEPA Prompt Optimizer for MATH")
    
    # Task Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model for the task (same as GRPO)")
    
    # vLLM (same as GRPO)
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM server for fast generation")
    parser.add_argument("--vllm_host", type=str, default="localhost")
    parser.add_argument("--vllm_port", type=int, default=8000)
    
    # Optimizer Model (OpenAI)
    parser.add_argument("--optimizer_model", type=str, default="gpt-4.1",
                        help="OpenAI model for reflection/prompt proposal")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="gepa_run")
    
    # GEPA parameters
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--minibatch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--candidate_selection", type=str, default="pareto",
                        choices=["pareto", "best"])
    
    # Evaluation (matching GRPO)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--num_completions_eval", type=int, default=20)
    parser.add_argument("--pass_at_k", type=int, default=1)
    
    # Dataset (matching GRPO)
    parser.add_argument("--train_size", type=int, default=12000)
    parser.add_argument("--eval_size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7111994)
    
    # Logging
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose logging of optimization process")
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Setup
    utils.seed_everything(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    logger = GEPALogger(args.output_dir, verbose=args.verbose)
    
    logger.section("GEPA - Genetic-Pareto Prompt Optimizer")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Seed: {args.seed}")
    
    # ========================================================================
    # SETUP MODELS (same as GRPO)
    # ========================================================================
    logger.subsection("Loading Models")
    logger.info(f"Task model: {args.model_name}")
    model, tokenizer = llms.get_llm_tokenizer(args.model_name)
    
    # vLLM client using OpenAI-compatible API (no custom TRL endpoints needed)
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}/v1"
        vllm_client = OpenAI(base_url=base_url, api_key="dummy")  # vLLM doesn't need real API key
        logger.info(f"Using vLLM server at {args.vllm_host}:{args.vllm_port}")
    else:
        logger.info("Using local model for generation")
    
    # OpenAI client for optimizer
    openai_client = OpenAI()
    logger.info(f"Optimizer model: {args.optimizer_model} (OpenAI)")
    
    # Load dataset
    logger.subsection("Loading Dataset")
    train_ds, eval_ds = load_math_dataset(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
    )
    train_data = list(train_ds)
    eval_data = list(eval_ds)
    logger.info(f"Train size: {len(train_data)}, Eval size: {len(eval_data)}")
    
    # ========================================================================
    # SEED PROMPT (same as GRPO)
    # ========================================================================
    seed_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, "
        "then <answer>your answer</answer>."
    )
    
    logger.subsection("Seed Prompt")
    logger.info("Initial system prompt:")
    for line in seed_prompt.split('\n'):
        logger.info(f"   {line}")
    
    # Initialize candidate pool
    candidates = [Candidate(prompt=seed_prompt, created_at_step=0)]
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    run_log = {
        "args": vars(args),
        "models": {
            "task_model": args.model_name,
            "optimizer_model": args.optimizer_model,
            "use_vllm": args.use_vllm,
        },
        "seed_prompt": seed_prompt,
        "steps": {},
        "prompt_evolution": [],
    }
    
    run_log["prompt_evolution"].append({
        "step": 0,
        "prompt": seed_prompt,
        "parent_idx": None,
        "trigger": "seed",
    })
    
    # ========================================================================
    # INITIAL VALIDATION SET SCORES (batched for speed)
    # ========================================================================
    logger.subsection("Initial Seed Prompt Evaluation")
    logger.info(f"Evaluating seed prompt on {len(eval_data)} validation problems...")
    
    # Build all prompts at once
    val_prompts = []
    val_entries_info = []
    for entry in eval_data:
        question = format_math_problem(entry)
        messages = [
            {"role": "system", "content": seed_prompt},
            {"role": "user", "content": question},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        val_prompts.append(prompt_text)
        val_entries_info.append((entry, extract_math_answer(entry)))
    
    # Generate all completions in one batched call
    if args.use_vllm:
        completions = generate_batch_vllm(
            vllm_client, args.model_name, val_prompts,
            temperature=args.temperature, max_tokens=args.max_completion_length
        )
    else:
        completions = generate_batch_local(
            model, tokenizer, val_prompts,
            temperature=args.temperature, max_tokens=args.max_completion_length
        )
    
    # Score all completions
    seed_correct = 0
    for val_idx, (completion, (entry, target_answer)) in enumerate(zip(completions, val_entries_info)):
        extracted = utils.extract_answer(completion)
        format_ok = utils.check_format(completion) > 0
        
        if not format_ok:
            score = 0.0
        elif extracted:
            score = 1.0 if eval_ds.score_answer(answer=extracted, entry=entry) == 1.0 else 0.0
        else:
            score = 0.0
        
        candidates[0].val_scores[val_idx] = score
        seed_correct += int(score)
    
    logger.success(f"Seed prompt: {seed_correct}/{len(eval_data)} correct (avg score: {candidates[0].avg_score():.3f})")
    
    # ========================================================================
    # MAIN GEPA LOOP
    # ========================================================================
    logger.section("Starting GEPA Optimization Loop")
    logger.info(f"Total iterations: {args.num_iters}")
    logger.info(f"Minibatch size: {args.minibatch_size}")
    logger.info(f"Candidate selection: {args.candidate_selection}")
    logger.info(f"Eval every: {args.eval_every} steps")
    
    for step in tqdm(range(args.num_iters), desc="GEPA Optimization"):
        
        # ====================================================================
        # PERIODIC EVALUATION (same as GRPO - uses batched generation)
        # ====================================================================
        if step % args.eval_every == 0:
            best_idx = select_candidate_best(candidates)
            best_prompt = candidates[best_idx].prompt
            
            avg_pass_at_k, avg_format, eval_examples = run_evaluation(
                system_prompt=best_prompt,
                eval_ds=eval_ds,
                tokenizer=tokenizer,
                model=model,
                model_name=args.model_name,
                vllm_client=vllm_client,
                use_vllm=args.use_vllm,
                num_completions=args.num_completions_eval,
                pass_at_k=args.pass_at_k,
                temperature=args.temperature,
                max_tokens=args.max_completion_length,
            )
            
            logger.eval_results(step, avg_pass_at_k, args.pass_at_k, avg_format, best_idx)
            
            # Log
            if step not in run_log["steps"]:
                run_log["steps"][step] = {}
            
            run_log["steps"][step]["eval"] = {
                "examples": eval_examples,
                "metrics": {
                    f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                    "avg_format_reward": avg_format,
                    "num_eval_problems": len(eval_ds),
                    "best_candidate_idx": best_idx,
                    "num_candidates": len(candidates),
                },
                "best_prompt": best_prompt,
            }
            
            # Save eval summary (same format as GRPO)
            eval_summary_path = os.path.join(args.output_dir, "eval_summary.json")
            eval_summary = {}
            if os.path.exists(eval_summary_path):
                with open(eval_summary_path, "r") as f:
                    eval_summary = json.load(f)
            
            eval_summary[str(step)] = {
                f"pass_at_{args.pass_at_k}": avg_pass_at_k,
                "avg_format_reward": avg_format,
                "num_candidates": len(candidates),
            }
            
            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, indent=2)
            
            # Save best prompt
            with open(os.path.join(args.output_dir, "best_prompt.txt"), "w") as f:
                f.write(best_prompt)
        
        # ====================================================================
        # STEP 1: SELECT CANDIDATE
        # ====================================================================
        logger.step_header(step, args.num_iters, len(candidates))
        
        if args.candidate_selection == "pareto":
            parent_idx = select_candidate_pareto(candidates, rng)
            selection_method = "pareto"
        else:
            parent_idx = select_candidate_best(candidates)
            selection_method = "best"
        
        parent_prompt = candidates[parent_idx].prompt
        parent_avg = candidates[parent_idx].avg_score()
        
        logger.parent_selected(parent_idx, parent_prompt, parent_avg, selection_method)
        
        # ====================================================================
        # STEP 2: SAMPLE MINIBATCH
        # ====================================================================
        minibatch = rng.sample(train_data, min(args.minibatch_size, len(train_data)))
        logger.info(f"Sampled minibatch of {len(minibatch)} problems")
        
        # ====================================================================
        # STEP 3: EVALUATE PARENT ON MINIBATCH
        # ====================================================================
        old_scores, traces = evaluate_prompt_on_batch(
            system_prompt=parent_prompt,
            batch=minibatch,
            dataset=train_ds,
            tokenizer=tokenizer,
            model=model,
            model_name=args.model_name,
            vllm_client=vllm_client,
            use_vllm=args.use_vllm,
            temperature=args.temperature,
            max_tokens=args.max_completion_length,
        )
        old_sum = sum(old_scores)
        
        logger.minibatch_eval(old_scores, traces)
        
        # Skip if all correct
        if all(s >= 1.0 for s in old_scores):
            logger.info("All correct - skipping (no improvement possible)")
            continue
        
        # ====================================================================
        # STEP 4: REFLECT AND PROPOSE NEW PROMPT
        # ====================================================================
        reflective_dataset = build_reflective_dataset(traces)
        logger.reflective_dataset(reflective_dataset)
        
        new_prompt, meta_prompt = propose_new_prompt(
            current_prompt=parent_prompt,
            reflective_dataset=reflective_dataset,
            openai_client=openai_client,
            optimizer_model=args.optimizer_model,
            logger=logger,
        )
        
        if new_prompt.strip() == parent_prompt.strip():
            logger.warning("Optimizer returned same prompt - skipping")
            continue
        
        # ====================================================================
        # STEP 5: EVALUATE NEW PROMPT ON SAME MINIBATCH
        # ====================================================================
        logger.subsection("Evaluating Proposed Prompt")
        new_scores, _ = evaluate_prompt_on_batch(
            system_prompt=new_prompt,
            batch=minibatch,
            dataset=train_ds,
            tokenizer=tokenizer,
            model=model,
            model_name=args.model_name,
            vllm_client=vllm_client,
            use_vllm=args.use_vllm,
            temperature=args.temperature,
            max_tokens=args.max_completion_length,
        )
        new_sum = sum(new_scores)
        
        logger.new_prompt_eval(old_scores, new_scores)
        
        # ====================================================================
        # STEP 6: ACCEPT IF IMPROVED
        # ====================================================================
        if new_sum > old_sum:
            new_candidate = Candidate(
                prompt=new_prompt,
                parent_idx=parent_idx,
                created_at_step=step,
            )
            
            # Evaluate on full validation set (batched for speed)
            logger.info("Evaluating new prompt on full validation set...")
            
            # Build all prompts
            val_prompts = []
            val_entries_info = []
            for entry in eval_data:
                question = format_math_problem(entry)
                messages = [
                    {"role": "system", "content": new_prompt},
                    {"role": "user", "content": question},
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                val_prompts.append(prompt_text)
                val_entries_info.append((entry, extract_math_answer(entry)))
            
            # Batch generate
            if args.use_vllm:
                val_completions = generate_batch_vllm(
                    vllm_client, args.model_name, val_prompts,
                    temperature=args.temperature, max_tokens=args.max_completion_length
                )
            else:
                val_completions = generate_batch_local(
                    model, tokenizer, val_prompts,
                    temperature=args.temperature, max_tokens=args.max_completion_length
                )
            
            # Score
            val_correct = 0
            for val_idx, (completion, (entry, _)) in enumerate(zip(val_completions, val_entries_info)):
                extracted = utils.extract_answer(completion)
                format_ok = utils.check_format(completion) > 0
                if not format_ok:
                    score = 0.0
                elif extracted:
                    score = 1.0 if eval_ds.score_answer(answer=extracted, entry=entry) == 1.0 else 0.0
                else:
                    score = 0.0
                new_candidate.val_scores[val_idx] = score
                val_correct += int(score)
            
            candidates.append(new_candidate)
            
            logger.acceptance_decision(
                accepted=True,
                old_score=old_sum,
                new_score=new_sum,
                val_score=new_candidate.avg_score()
            )
            
            logger.info(f"New candidate #{len(candidates)-1}: {val_correct}/{len(eval_data)} correct on validation")
            logger.candidate_status(candidates)
            
            run_log["prompt_evolution"].append({
                "step": step,
                "prompt": new_prompt,
                "parent_idx": parent_idx,
                "parent_prompt": parent_prompt,
                "old_minibatch_score": old_sum,
                "new_minibatch_score": new_sum,
                "val_avg_score": new_candidate.avg_score(),
                "val_correct": val_correct,
                "val_total": len(eval_data),
                "reflective_dataset": reflective_dataset,
                "meta_prompt_sent": meta_prompt,
            })
        else:
            logger.acceptance_decision(
                accepted=False,
                old_score=old_sum,
                new_score=new_sum
            )
        
        # ====================================================================
        # LOG
        # ====================================================================
        if step not in run_log["steps"]:
            run_log["steps"][step] = {}
        
        run_log["steps"][step]["train"] = {
            "parent_idx": parent_idx,
            "parent_prompt_preview": parent_prompt[:200] + "..." if len(parent_prompt) > 200 else parent_prompt,
            "old_minibatch_score": old_sum,
            "new_minibatch_score": new_sum,
            "proposed_prompt_preview": new_prompt[:200] + "..." if len(new_prompt) > 200 else new_prompt,
            "accepted": new_sum > old_sum,
            "num_candidates": len(candidates),
        }
        
        # Save more frequently when verbose
        if step % 5 == 0 or (new_sum > old_sum):
            with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
                json.dump(run_log, f, indent=2)
    
    # ========================================================================
    # FINAL SAVE
    # ========================================================================
    logger.section("GEPA Optimization Complete")
    
    best_idx = select_candidate_best(candidates)
    best_prompt = candidates[best_idx].prompt
    
    with open(os.path.join(args.output_dir, "best_prompt.txt"), "w") as f:
        f.write(best_prompt)
    
    all_prompts = [
        {
            "idx": i,
            "prompt": c.prompt,
            "parent_idx": c.parent_idx,
            "created_at_step": c.created_at_step,
            "avg_val_score": c.avg_score(),
        }
        for i, c in enumerate(candidates)
    ]
    
    with open(os.path.join(args.output_dir, "all_candidates.json"), "w") as f:
        json.dump(all_prompts, f, indent=2)
    
    run_log["final"] = {
        "num_candidates": len(candidates),
        "best_candidate_idx": best_idx,
        "best_avg_score": candidates[best_idx].avg_score(),
    }
    
    with open(os.path.join(args.output_dir, "run_log.json"), "w") as f:
        json.dump(run_log, f, indent=2)
    
    # Final summary
    logger.success(f"Total candidates evolved: {len(candidates)}")
    logger.success(f"Best candidate: #{best_idx} (avg score: {candidates[best_idx].avg_score():.3f})")
    
    logger.subsection("Best Prompt")
    for line in best_prompt.split('\n'):
        logger.info(f"   {line}")
    
    logger.subsection("Output Files")
    logger.info(f"   Best prompt: {args.output_dir}/best_prompt.txt")
    logger.info(f"   All candidates: {args.output_dir}/all_candidates.json")
    logger.info(f"   Eval summary: {args.output_dir}/eval_summary.json")
    logger.info(f"   Full run log: {args.output_dir}/run_log.json")
    logger.info(f"   Optimization log: {args.output_dir}/optimization_log.txt")
    
    logger.candidate_status(candidates)


if __name__ == "__main__":
    main()
