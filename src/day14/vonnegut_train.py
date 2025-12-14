"""
vonnegut_train.py - Two-Loop Learning: Text Skill Refinement + Cartridge Condensation

This implements a continual learning system with two clocks:
1. Fast Clock (Skill Loop): Evolves a skill.md file via LLM feedback
2. Slow Clock (Cartridge Loop): Condenses the skill into learnable KV cache vectors

The hypothesis: The cartridge can become a "better representation" than the text,
finding activation patterns that natural language can't easily describe.

Usage:
    python vonnegut_train.py --output my_run --iterations 500
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# The Vonnegut rubric for judging
VONNEGUT_RUBRIC = """
Score each dimension from 1-10:

1. SENTENCE_SIMPLICITY: Vonnegut uses short, punchy sentences. Complex or flowery prose is bad.
2. FATALISM: Does it have resigned, humanist fatalism? "So it goes" energy.
3. DARK_HUMOR: Is it funny in a bleak, satirical way?
4. DIRECTNESS: Does it say things plainly without pretension?
5. HUMANITY: Does it show compassion for human weakness and folly?

Output as JSON: {"sentence_simplicity": X, "fatalism": X, "dark_humor": X, "directness": X, "humanity": X, "total": X}
where total = average of all scores.
"""

# Starting skill (intentionally minimal)
INITIAL_SKILL = """Write in the style of Kurt Vonnegut."""

# 50 writing subjects (40 train, 10 eval)
SUBJECTS = [
    # Training subjects (0-39)
    "a birthday party for a 90-year-old",
    "waiting in line at the DMV",
    "a first date at a fast food restaurant",
    "attending a funeral for someone you didn't like",
    "watching the moon landing on TV",
    "a job interview for a position you don't want",
    "finding a $20 bill on the sidewalk",
    "being stuck in an elevator with a stranger",
    "a high school reunion after 30 years",
    "watching your childhood home get demolished",
    "a conversation with a telemarketer",
    "the last day of a dying shopping mall",
    "teaching a child about death",
    "a soldier's last letter home",
    "waiting for medical test results",
    "a wedding where the bride doesn't show up",
    "cleaning out a dead parent's house",
    "a robot learning to feel loneliness",
    "the inventor of the atomic bomb at a dinner party",
    "a time traveler stuck in the wrong decade",
    "a billionaire's existential crisis",
    "the last bookstore in a digital world",
    "a retired executioner's memoir",
    "aliens observing human traffic jams",
    "a dog's understanding of human sadness",
    "the janitor at a nuclear power plant",
    "a greeting card writer who hates sentiment",
    "the last person to remember World War I",
    "a comedian performing at a hospice",
    "finding your own obituary in the newspaper",
    "a AI therapist developing anxiety",
    "the night shift at a 24-hour diner",
    "a protest that nobody attends",
    "the guy who has to tell people their flight is cancelled",
    "a love letter written by committee",
    "the invention of bureaucracy",
    "a prayer from an atheist",
    "the last typewriter repairman",
    "a self-driving car's existential monologue",
    "the person who writes fortune cookies",
    # Eval subjects (40-49)
    "a graduation speech for a school that's closing",
    "the zoo after humans are gone",
    "a vending machine's view of humanity",
    "the receptionist at the end of the world",
    "a time capsule opened too early",
    "the inventor of the snooze button",
    "a support group for retired superheroes",
    "the last ice cream truck driver",
    "a robot reading poetry for the first time",
    "the people who paint highway lines",
]

TRAIN_SUBJECTS = SUBJECTS[:40]
EVAL_SUBJECTS = SUBJECTS[40:]


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_name):
    """Load the model and tokenizer."""
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Extract model dimensions
    config = model.config
    model_info = {
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
    }
    
    print(f"  Layers: {model_info['num_layers']}")
    print(f"  KV Heads: {model_info['num_kv_heads']}")
    print(f"  Head Dim: {model_info['head_dim']}")
    
    return model, tokenizer, model_info


# =============================================================================
# CARTRIDGE OPERATIONS
# =============================================================================

def create_empty_cartridge(model_info):
    """Create an empty cartridge (no tokens yet)."""
    return {
        "keys": [],      # List of [num_layers, num_kv_heads, num_tokens, head_dim] tensors
        "values": [],    # List of [num_layers, num_kv_heads, num_tokens, head_dim] tensors
        "num_tokens": 0,
    }


def get_kv_cache_from_text(model, tokenizer, text, num_tokens):
    """
    Run text through the model and extract the first num_tokens of the KV cache.
    This gives us a "warm start" for cartridge training.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # Pad or truncate to num_tokens
    if len(tokens) < num_tokens:
        tokens = tokens * ((num_tokens // len(tokens)) + 1)
    tokens = tokens[:num_tokens]
    
    input_ids = torch.tensor([tokens], device=DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
    
    # Extract keys and values: shape [num_layers, num_kv_heads, num_tokens, head_dim]
    keys = torch.stack([layer[0][0] for layer in past_kv], dim=0)
    values = torch.stack([layer[1][0] for layer in past_kv], dim=0)
    
    return keys, values


def add_trainable_tokens_to_cartridge(cartridge, new_keys, new_values):
    """Add new trainable tokens to the cartridge."""
    cartridge["keys"].append(new_keys)
    cartridge["values"].append(new_values)
    cartridge["num_tokens"] += new_keys.shape[2]
    return cartridge


def get_cartridge_tensors(cartridge):
    """Get the full cartridge as concatenated tensors."""
    if cartridge["num_tokens"] == 0:
        return None, None
    
    keys = torch.cat(cartridge["keys"], dim=2)
    values = torch.cat(cartridge["values"], dim=2)
    return keys, values


def freeze_cartridge(cartridge):
    """Freeze all current cartridge parameters (detach and make non-trainable)."""
    for i in range(len(cartridge["keys"])):
        cartridge["keys"][i] = cartridge["keys"][i].detach().clone()
        cartridge["values"][i] = cartridge["values"][i].detach().clone()
    return cartridge


# =============================================================================
# GENERATION WITH CARTRIDGE
# =============================================================================

def generate_with_cartridge(model, tokenizer, model_info, prompt, skill_text, cartridge, 
                            max_tokens=200, temperature=0.8):
    """
    Generate text using the model with:
    - Cartridge KV cache prepended
    - Skill text in the context
    """
    # Build the full prompt with skill
    full_prompt = f"""You are a writer. Follow these style instructions:

{skill_text}

Now write about: {prompt}

Write a single paragraph (3-6 sentences) in this style:"""
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    # Get cartridge tensors
    cart_keys, cart_values = get_cartridge_tensors(cartridge)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Build KV cache with cartridge prefix
            if cart_keys is not None:
                cache = DynamicCache()
                for layer_idx in range(model_info["num_layers"]):
                    cache.update(
                        cart_keys[layer_idx].unsqueeze(0),
                        cart_values[layer_idx].unsqueeze(0),
                        layer_idx,
                    )
                
                cart_len = cart_keys.shape[2]
                attn_mask = torch.ones(1, cart_len + input_ids.shape[1], device=DEVICE)
                position_ids = torch.arange(input_ids.shape[1], device=DEVICE).unsqueeze(0) + cart_len
            else:
                cache = None
                attn_mask = None
                position_ids = None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated_ids.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_samples(model, tokenizer, model_info, prompt, skill_text, cartridge, n_samples=4):
    """Generate n samples for a given prompt."""
    samples = []
    for _ in range(n_samples):
        sample = generate_with_cartridge(
            model, tokenizer, model_info, prompt, skill_text, cartridge
        )
        samples.append(sample)
    return samples


# =============================================================================
# GPT-4.1 JUDGE AND SKILL UPDATER
# =============================================================================

def call_openai(messages, model="gpt-4.1", temperature=0.7, max_tokens=2000):
    """Call OpenAI API."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def judge_sample(sample, prompt):
    """Have GPT-4.1 score a single sample using the Vonnegut rubric."""
    judge_prompt = f"""You are judging writing style. The writer was asked to write about: "{prompt}"

They wrote:
---
{sample}
---

{VONNEGUT_RUBRIC}

Respond with ONLY the JSON scores, nothing else."""
    
    messages = [{"role": "user", "content": judge_prompt}]
    response = call_openai(messages, model="gpt-4.1", temperature=0.3)
    
    # Parse the JSON
    try:
        # Handle potential markdown code blocks
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        scores = json.loads(response.strip())
    except json.JSONDecodeError:
        # Fallback scores if parsing fails
        scores = {"sentence_simplicity": 5, "fatalism": 5, "dark_humor": 5, 
                  "directness": 5, "humanity": 5, "total": 5}
    
    return scores


def judge_samples(samples, prompt):
    """Judge multiple samples, return list of scores."""
    return [judge_sample(sample, prompt) for sample in samples]


def update_skill(current_skill, samples_with_scores, prompt, max_skill_tokens=512, tokenizer=None):
    """
    Have GPT-4.1 update the skill.md based on the samples and their scores.
    Returns the new skill text, constrained to max_skill_tokens.
    """
    # Format samples with their scores
    samples_text = ""
    for i, (sample, scores) in enumerate(samples_with_scores):
        samples_text += f"\n--- Sample {i+1} (total score: {scores.get('total', 'N/A')}) ---\n"
        samples_text += f"{sample}\n"
        samples_text += f"Scores: {json.dumps(scores)}\n"
    
    update_prompt = f"""You are helping improve a writing style instruction file.

The current style instructions are:
---
{current_skill}
---

The writer was asked to write about: "{prompt}"

Here are their outputs and scores (1-10 scale, higher is better):
{samples_text}

Based on these results, please EDIT the style instructions to help the writer improve.

IMPORTANT CONSTRAINTS:
- Keep instructions CONCISE - maximum ~{max_skill_tokens} tokens (roughly 400 words)
- Focus on the MOST IMPORTANT rules only
- Remove redundant or low-impact instructions
- Be specific but brief

Output ONLY the new style instructions, nothing else:"""

    messages = [{"role": "user", "content": update_prompt}]
    new_skill = call_openai(messages, model="gpt-4.1", temperature=0.7)
    new_skill = new_skill.strip()
    
    # Hard truncate if still too long (safety net)
    if tokenizer is not None:
        tokens = tokenizer.encode(new_skill, add_special_tokens=False)
        if len(tokens) > max_skill_tokens:
            # Truncate and add note
            tokens = tokens[:max_skill_tokens]
            new_skill = tokenizer.decode(tokens)
            # Try to end at a sentence
            last_period = new_skill.rfind('.')
            if last_period > len(new_skill) // 2:
                new_skill = new_skill[:last_period + 1]
    
    return new_skill


def generate_baseline_sample_external(prompt, baseline_model="gpt-4.1-nano"):
    """Generate a Vonnegut-style sample using an external API model."""
    gen_prompt = f"""Write a single paragraph (3-6 sentences) in the style of Kurt Vonnegut about: {prompt}

Channel Vonnegut's voice: short sentences, dark humor, resigned fatalism, plain speaking, 
and deep compassion for human folly. Include his characteristic rhythm and perhaps 
a phrase like "So it goes" if it fits naturally."""

    messages = [{"role": "user", "content": gen_prompt}]
    return call_openai(messages, model=baseline_model, temperature=0.8, max_tokens=300)


def generate_baseline_sample_self(model, tokenizer, model_info, prompt):
    """Generate a sample using the same model but WITHOUT skill or cartridge."""
    # Empty cartridge and minimal prompt (no skill instructions)
    empty_cartridge = create_empty_cartridge(model_info)
    no_skill = "Write in the style of Kurt Vonnegut."  # Just the basic instruction
    
    return generate_with_cartridge(
        model, tokenizer, model_info, prompt, no_skill, empty_cartridge,
        max_tokens=200, temperature=0.8
    )


def generate_baseline_sample_previous(model, tokenizer, model_info, prompt, prev_skill, prev_cartridge):
    """Generate a sample using the model with the PREVIOUS iteration's skill and cartridge."""
    return generate_with_cartridge(
        model, tokenizer, model_info, prompt, prev_skill, prev_cartridge,
        max_tokens=200, temperature=0.8
    )


def generate_baseline_samples(prompt, n_samples=2, baseline_mode="self", baseline_model="gpt-4.1-nano",
                               model=None, tokenizer=None, model_info=None,
                               prev_skill=None, prev_cartridge=None):
    """Generate n baseline samples. 
    
    baseline_mode: "self" = same model without skill/cartridge
                   "external" = API model
                   "previous" = same model with previous iteration's skill/cartridge
    """
    samples = []
    for _ in range(n_samples):
        if baseline_mode == "self":
            sample = generate_baseline_sample_self(model, tokenizer, model_info, prompt)
        elif baseline_mode == "previous":
            sample = generate_baseline_sample_previous(model, tokenizer, model_info, prompt, prev_skill, prev_cartridge)
        else:
            sample = generate_baseline_sample_external(prompt, baseline_model)
        samples.append(sample)
    return samples


# =============================================================================
# ROUND ROBIN TOURNAMENT EVALUATION
# =============================================================================

def compare_pair(sample_a, sample_b, prompt):
    """Have GPT-4.1 judge which sample is more Vonnegut-like. Returns 'A', 'B', or 'TIE'."""
    compare_prompt = f"""You are judging which text better captures Kurt Vonnegut's writing style.
The topic was: "{prompt}"

Text A:
---
{sample_a}
---

Text B:
---
{sample_b}
---

Which text better captures Vonnegut's style? Consider:
- Short, punchy sentences
- Dark humor and satire
- Resigned, humanist fatalism
- Plain, direct language
- Compassion for human weakness

Reply with ONLY one of: A, B, or TIE"""

    messages = [{"role": "user", "content": compare_prompt}]
    response = call_openai(messages, model="gpt-4.1", temperature=0.3, max_tokens=10)
    
    response = response.strip().upper()
    if "A" in response and "B" not in response:
        return "A"
    elif "B" in response and "A" not in response:
        return "B"
    else:
        return "TIE"


def direct_comparison_eval(our_samples, baseline_samples, prompt):
    """
    Direct 1:1 comparison (sample i vs sample i).
    Much faster than round-robin.
    Returns (our_wins, total_matchups, matchup_details).
    """
    our_wins = 0
    total = 0
    matchups = []
    
    n = min(len(our_samples), len(baseline_samples))
    
    for i in range(n):
        result = compare_pair(our_samples[i], baseline_samples[i], prompt)
        
        matchups.append({
            "idx": i,
            "result": result,
        })
        
        total += 1
        if result == "A":  # Our sample won
            our_wins += 1
        elif result == "TIE":
            our_wins += 0.5
    
    return our_wins, total, matchups


def run_full_eval(model, tokenizer, model_info, skill_text, cartridge, eval_subjects,
                  n_samples=2, baseline_mode="self", baseline_model="gpt-4.1-nano", num_eval_subjects=5,
                  prev_skill=None, prev_cartridge=None):
    """
    Run evaluation on eval subjects.
    Uses direct 1:1 comparison (not round-robin) for speed.
    
    baseline_mode: "self" = compare to same model without skill/cartridge
                   "external" = compare to external API model
                   "previous" = compare to previous iteration's skill/cartridge
    """
    if baseline_mode == "previous":
        baseline_desc = "previous iteration"
    elif baseline_mode == "external":
        baseline_desc = baseline_model
    else:
        baseline_desc = "self (no skill/cartridge)"
    
    results = {
        "subjects": [],
        "total_wins": 0,
        "total_matchups": 0,
        "baseline_mode": baseline_mode,
        "baseline_model": baseline_desc,
    }
    
    # Limit number of subjects
    subjects_to_eval = eval_subjects[:num_eval_subjects]
    
    for subject in subjects_to_eval:
        print(f"    Evaluating: {subject[:40]}...")
        
        # Generate samples from our model (with skill + cartridge)
        our_samples = generate_samples(
            model, tokenizer, model_info, subject, skill_text, cartridge, n_samples=n_samples
        )
        
        # Generate samples from baseline
        baseline_samples = generate_baseline_samples(
            subject, n_samples=n_samples, baseline_mode=baseline_mode, baseline_model=baseline_model,
            model=model, tokenizer=tokenizer, model_info=model_info,
            prev_skill=prev_skill, prev_cartridge=prev_cartridge
        )
        
        # Direct 1:1 comparison
        wins, total, matchups = direct_comparison_eval(our_samples, baseline_samples, subject)
        
        results["subjects"].append({
            "subject": subject,
            "our_samples": our_samples,
            "baseline_samples": baseline_samples,
            "wins": wins,
            "total": total,
            "win_rate": wins / total if total > 0 else 0,
            "matchups": matchups,
        })
        
        results["total_wins"] += wins
        results["total_matchups"] += total
    
    results["overall_win_rate"] = results["total_wins"] / results["total_matchups"] if results["total_matchups"] > 0 else 0
    
    return results


# =============================================================================
# CARTRIDGE TRAINING (CONTEXT DISTILLATION)
# =============================================================================

def train_cartridge_step(model, tokenizer, model_info, trainable_keys, trainable_values,
                         frozen_cartridge, skill_text, prompt, optimizer):
    """
    One step of cartridge training via context distillation.
    
    Teacher: model + skill_text in context
    Student: model + cartridge (frozen + trainable)
    
    Returns the loss value.
    """
    # Build the question prompt (same structure, just shorter)
    question = f"Write about: {prompt}\n\nWrite a single paragraph:"
    
    # Generate teacher's answer and get top-k logprobs
    teacher_messages = [
        {"role": "system", "content": f"Follow these style instructions:\n{skill_text}"},
        {"role": "user", "content": question}
    ]
    teacher_text = tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
    
    # Generate a short answer with the teacher
    teacher_input_ids = tokenizer(teacher_text, return_tensors="pt").input_ids.to(DEVICE)
    
    with torch.no_grad():
        teacher_output = model.generate(
            teacher_input_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer_ids = teacher_output[0, teacher_input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    # Get teacher's logprobs for this answer
    teacher_full_ids = teacher_output
    teacher_prompt_len = teacher_input_ids.shape[1]
    
    with torch.no_grad():
        teacher_out = model(input_ids=teacher_full_ids, use_cache=False)
        teacher_logits = teacher_out.logits[0, teacher_prompt_len-1:-1, :]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Get top-20 for sparse CE
        top_k = 20
        topk_probs, topk_ids = torch.topk(teacher_probs, k=top_k, dim=-1)
    
    # Now run student (with cartridge)
    student_messages = [{"role": "user", "content": question}]
    student_text = tokenizer.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True)
    student_full_text = student_text + answer_text
    student_tokens = tokenizer(student_full_text, return_tensors="pt").input_ids.to(DEVICE)
    student_prompt_len = len(tokenizer(student_text).input_ids)
    
    # Build student's KV cache with cartridge
    frozen_keys, frozen_values = get_cartridge_tensors(frozen_cartridge)
    
    # Concatenate frozen + trainable cartridge
    if frozen_keys is not None:
        full_keys = torch.cat([frozen_keys, trainable_keys], dim=2)
        full_values = torch.cat([frozen_values, trainable_values], dim=2)
    else:
        full_keys = trainable_keys
        full_values = trainable_values
    
    cart_len = full_keys.shape[2]
    
    # Build cache
    cache = DynamicCache()
    for layer_idx in range(model_info["num_layers"]):
        cache.update(
            full_keys[layer_idx].unsqueeze(0),
            full_values[layer_idx].unsqueeze(0),
            layer_idx,
        )
    
    attn_mask = torch.ones(1, cart_len + student_tokens.shape[1], device=DEVICE)
    position_ids = torch.arange(student_tokens.shape[1], device=DEVICE).unsqueeze(0) + cart_len
    
    # Student forward pass
    student_out = model(
        input_ids=student_tokens,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    
    # Get student logprobs for answer positions
    student_logits = student_out.logits[0, student_prompt_len-1:-1, :]
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Align lengths
    num_tokens = min(student_logits.shape[0], topk_probs.shape[0])
    
    # Sparse top-k cross-entropy
    student_topk_logprobs = student_log_probs[:num_tokens].gather(
        dim=-1, index=topk_ids[:num_tokens]
    )
    
    ce_by_token = -(topk_probs[:num_tokens] * student_topk_logprobs).sum(dim=-1)
    loss = ce_by_token.mean()
    
    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item(), answer_text


# =============================================================================
# LOGGING
# =============================================================================

def save_json(data, path):
    """Save data as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def save_text(text, path):
    """Save text to file."""
    with open(path, "w") as f:
        f.write(text)


def setup_logging(output_dir):
    """Create logging directories."""
    dirs = [
        output_dir,
        output_dir / "skills",
        output_dir / "training_logs",
        output_dir / "cartridge_logs", 
        output_dir / "eval_logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vonnegut Style Learning: Text + Cartridge")
    parser.add_argument("--output", type=str, default="vonnegut_run", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--iterations", type=int, default=500, help="Number of outer loop iterations")
    parser.add_argument("--skill-rounds", type=int, default=10, help="Skill refinement rounds per iteration")
    parser.add_argument("--cartridge-steps", type=int, default=50, help="Cartridge training steps per iteration")
    parser.add_argument("--tokens-per-iter", type=int, default=32, help="New cartridge tokens per iteration")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for cartridge training")
    parser.add_argument("--samples-per-prompt", type=int, default=2, help="Samples to generate per prompt")
    parser.add_argument("--max-skill-tokens", type=int, default=512, help="Max tokens for skill file (prevents bloat)")
    # Eval settings
    parser.add_argument("--baseline-mode", type=str, default="self", choices=["self", "external", "previous"],
                        help="Baseline: 'self' = no skill/cartridge, 'external' = API model, 'previous' = previous iteration")
    parser.add_argument("--baseline-model", type=str, default="gpt-4.1-nano", 
                        help="External baseline model (only used if --baseline-mode=external)")
    parser.add_argument("--eval-every", type=int, default=1, help="Run eval every N iterations (1=every iteration)")
    parser.add_argument("--num-eval-subjects", type=int, default=5, help="Number of eval subjects to use")
    parser.add_argument("--eval-samples", type=int, default=2, help="Samples per subject for eval")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    setup_logging(output_dir)
    
    print("=" * 70)
    print("VONNEGUT STYLE LEARNING")
    print("Text Skill Refinement + Cartridge Condensation")
    print("=" * 70)
    
    # Save config
    save_json(vars(args), output_dir / "config.json")
    
    # Load model
    model, tokenizer, model_info = load_model(args.model)
    
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    # Initialize
    current_skill = INITIAL_SKILL
    cartridge = create_empty_cartridge(model_info)
    
    # Save initial skill
    save_text(current_skill, output_dir / "skills" / "skill_iter_0.md")
    
    # Track metrics over time
    metrics_history = []
    
    print(f"\nStarting training for {args.iterations} iterations")
    print(f"  Skill rounds per iteration: {args.skill_rounds}")
    print(f"  Max skill tokens: {args.max_skill_tokens}")
    print(f"  Cartridge steps per iteration: {args.cartridge_steps}")
    print(f"  New cartridge tokens per iteration: {args.tokens_per_iter}")
    if args.baseline_mode == "self":
        print(f"  Baseline: same model without skill/cartridge")
    elif args.baseline_mode == "previous":
        print(f"  Baseline: previous iteration's skill/cartridge")
    else:
        print(f"  Baseline: {args.baseline_model}")
    print(f"  Eval every: {args.eval_every} iterations")
    print(f"  Eval subjects: {args.num_eval_subjects}, samples: {args.eval_samples}")
    print()
    
    # Track previous skill/cartridge for "previous" baseline mode
    prev_skill = INITIAL_SKILL
    prev_cartridge = create_empty_cartridge(model_info)
    
    # Initial eval before any training (always vs base model for initial)
    baseline_desc = "base model" if args.baseline_mode in ["self", "previous"] else args.baseline_model
    print(f"\n[INITIAL EVAL] Before training starts (vs {baseline_desc})...")
    initial_eval = run_full_eval(
        model, tokenizer, model_info, current_skill, cartridge, EVAL_SUBJECTS,
        n_samples=args.eval_samples, baseline_mode="self",  # Always use self for initial
        baseline_model=args.baseline_model, num_eval_subjects=args.num_eval_subjects
    )
    save_json(initial_eval, output_dir / "eval_logs" / "initial_eval.json")
    print(f"  Initial win rate: {initial_eval['overall_win_rate']:.1%}")
    
    # Track initial in metrics
    metrics_history.append({
        "iteration": 0,
        "cartridge_tokens": 0,
        "win_rate_before": None,
        "win_rate_after_skill": None,
        "win_rate_after_cartridge": initial_eval["overall_win_rate"],
        "final_loss": None,
        "time_seconds": 0,
        "did_eval": True,
    })
    save_json(metrics_history, output_dir / "metrics_history.json")
    
    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{args.iterations}")
        print(f"{'='*70}")
        
        iter_log = {
            "iteration": iteration,
            "skill_rounds": [],
            "cartridge_training": [],
            "evals": {},
        }
        
        # Reset skill to simple starting point each iteration
        # The cartridge accumulates knowledge; the skill is refined fresh each time
        current_skill = INITIAL_SKILL
        
        # Check if we should eval this iteration
        do_eval = (iteration % args.eval_every == 0)
        
        # =====================================================================
        # PHASE A: Skill Refinement (10 rounds)
        # =====================================================================
        print(f"\n[PHASE A] Skill Refinement ({args.skill_rounds} rounds)")
        
        for round_idx in range(args.skill_rounds):
            # Sample a random training subject
            subject = random.choice(TRAIN_SUBJECTS)
            
            print(f"  Round {round_idx + 1}: {subject[:40]}...")
            
            # Generate samples
            samples = generate_samples(
                model, tokenizer, model_info, subject, current_skill, cartridge,
                n_samples=args.samples_per_prompt
            )
            
            # Judge samples
            scores = judge_samples(samples, subject)
            
            # Combine for logging
            samples_with_scores = list(zip(samples, scores))
            
            # Calculate average score for this round
            avg_score = sum(s.get("total", 5) for s in scores) / len(scores)
            
            # Update skill
            old_skill = current_skill
            current_skill = update_skill(current_skill, samples_with_scores, subject, 
                                         max_skill_tokens=args.max_skill_tokens, tokenizer=tokenizer)
            
            # Log this round
            round_log = {
                "round": round_idx + 1,
                "subject": subject,
                "samples": samples,
                "scores": scores,
                "avg_score": avg_score,
                "old_skill": old_skill,
                "new_skill": current_skill,
            }
            iter_log["skill_rounds"].append(round_log)
            
            print(f"    Avg score: {avg_score:.1f}/10")
        
        # Save skill after Phase A
        save_text(current_skill, output_dir / "skills" / f"skill_iter_{iteration}_after_phase_a.md")
        
        # =====================================================================
        # EVAL: After Phase A
        # =====================================================================
        if do_eval:
            print("\n[EVAL] After skill refinement...")
            eval_after_skill = run_full_eval(
                model, tokenizer, model_info, current_skill, cartridge, EVAL_SUBJECTS,
                n_samples=args.eval_samples, baseline_mode=args.baseline_mode,
                baseline_model=args.baseline_model, num_eval_subjects=args.num_eval_subjects,
                prev_skill=prev_skill, prev_cartridge=prev_cartridge
            )
            iter_log["evals"]["after_skill"] = eval_after_skill
            print(f"  Win rate: {eval_after_skill['overall_win_rate']:.1%}")
        else:
            eval_after_skill = None
        
        # =====================================================================
        # PHASE B: Cartridge Training
        # =====================================================================
        print(f"\n[PHASE B] Cartridge Training ({args.cartridge_steps} steps)")
        
        # Freeze current cartridge
        cartridge = freeze_cartridge(cartridge)
        
        # Initialize new trainable tokens from skill's KV cache
        new_keys, new_values = get_kv_cache_from_text(
            model, tokenizer, current_skill, args.tokens_per_iter
        )
        
        # Make them trainable
        trainable_keys = torch.nn.Parameter(new_keys.clone())
        trainable_values = torch.nn.Parameter(new_values.clone())
        
        optimizer = torch.optim.Adam([trainable_keys, trainable_values], lr=args.lr)
        
        losses = []
        for step in range(args.cartridge_steps):
            # Sample a training subject
            subject = random.choice(TRAIN_SUBJECTS)
            
            loss, answer = train_cartridge_step(
                model, tokenizer, model_info,
                trainable_keys, trainable_values,
                cartridge, current_skill, subject, optimizer
            )
            
            losses.append(loss)
            
            iter_log["cartridge_training"].append({
                "step": step + 1,
                "subject": subject,
                "loss": loss,
            })
            
            if (step + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / 10
                print(f"  Step {step + 1}: loss = {avg_loss:.4f}")
        
        # Add trained tokens to cartridge
        cartridge = add_trainable_tokens_to_cartridge(
            cartridge, trainable_keys.detach(), trainable_values.detach()
        )
        
        print(f"  Cartridge now has {cartridge['num_tokens']} tokens")
        
        # =====================================================================
        # EVAL: After Phase B
        # =====================================================================
        if do_eval:
            print("\n[EVAL] After cartridge training...")
            eval_after_cartridge = run_full_eval(
                model, tokenizer, model_info, current_skill, cartridge, EVAL_SUBJECTS,
                n_samples=args.eval_samples, baseline_mode=args.baseline_mode,
                baseline_model=args.baseline_model, num_eval_subjects=args.num_eval_subjects,
                prev_skill=prev_skill, prev_cartridge=prev_cartridge
            )
            iter_log["evals"]["after_cartridge"] = eval_after_cartridge
            print(f"  Win rate: {eval_after_cartridge['overall_win_rate']:.1%}")
        else:
            eval_after_cartridge = None
        
        # =====================================================================
        # Save iteration logs
        # =====================================================================
        save_json(iter_log, output_dir / "training_logs" / f"iter_{iteration:04d}.json")
        save_text(current_skill, output_dir / "skills" / f"skill_iter_{iteration}.md")
        
        # Save cartridge checkpoint
        torch.save({
            "keys": cartridge["keys"],
            "values": cartridge["values"],
            "num_tokens": cartridge["num_tokens"],
        }, output_dir / "cartridge_logs" / f"cartridge_iter_{iteration:04d}.pt")
        
        # Track metrics
        metrics = {
            "iteration": iteration,
            "cartridge_tokens": cartridge["num_tokens"],
            "win_rate_after_skill": eval_after_skill["overall_win_rate"] if eval_after_skill else None,
            "win_rate_after_cartridge": eval_after_cartridge["overall_win_rate"] if eval_after_cartridge else None,
            "final_loss": losses[-1] if losses else None,
            "time_seconds": time.time() - iter_start,
            "did_eval": do_eval,
        }
        metrics_history.append(metrics)
        
        # Save metrics
        save_json(metrics_history, output_dir / "metrics_history.json")
        
        print(f"\n  Iteration complete in {metrics['time_seconds']:.1f}s")
        if do_eval:
            print(f"  Win rates: skill={metrics['win_rate_after_skill']:.1%}, cartridge={metrics['win_rate_after_cartridge']:.1%}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final skill saved to: {output_dir}/skills/skill_iter_{args.iterations}.md")
    print(f"Final cartridge: {cartridge['num_tokens']} tokens")
    print(f"Metrics history: {output_dir}/metrics_history.json")
    
    # Print win rate progression
    print("\nWin rate progression (after cartridge):")
    for m in metrics_history[::max(1, len(metrics_history)//10)]:  # Sample 10 points
        print(f"  Iter {m['iteration']:3d}: {m['win_rate_after_cartridge']:.1%}")


if __name__ == "__main__":
    main()

