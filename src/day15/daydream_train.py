"""
daydream_train.py - ENGRAM for Creative Concept Linking (Daydreaming Loop)

This implements a continual learning system for finding non-obvious connections
between concepts - inspired by the "daydreaming" hypothesis for AI insight.

The task: Given two random concepts, find a novel, coherent, deep connection.

Two clocks:
1. Fast Clock (Skill Loop): Evolves a skill.md file via LLM feedback
2. Slow Clock (Cartridge Loop): Condenses the skill into learnable KV cache vectors

Usage:
    python daydream_train.py --output my_run --iterations 100
"""

import argparse
import json
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

# Rubric for judging concept connections
CONNECTION_RUBRIC = """
Score each dimension from 1-10:

1. NOVELTY: Is this a connection most people wouldn't think of? (Not obvious or cliché)
2. COHERENCE: Does the connection actually make logical sense? (Not forced or nonsensical)
3. DEPTH: Is this a deep structural insight, or just surface-level wordplay?
4. GENERATIVITY: Does this connection spark further ideas or questions?

Output as JSON: {"novelty": X, "coherence": X, "depth": X, "generativity": X, "total": X}
where total = average of all scores.
"""

# Starting skill (intentionally minimal)
INITIAL_SKILL = """Find interesting and non-obvious connections between concepts."""

# 50 concepts for pairing (mix of philosophical, scientific, abstract, concrete)
CONCEPTS = [
    # Philosophical (0-9)
    "consciousness",
    "free will",
    "meaning",
    "death",
    "identity",
    "truth",
    "beauty",
    "justice",
    "time",
    "existence",
    
    # Scientific (10-19)
    "entropy",
    "evolution",
    "emergence",
    "gravity",
    "quantum superposition",
    "DNA",
    "neural networks",
    "ecosystems",
    "black holes",
    "photosynthesis",
    
    # Abstract (20-29)
    "paradox",
    "infinity",
    "chaos",
    "symmetry",
    "recursion",
    "boundaries",
    "change",
    "memory",
    "language",
    "creativity",
    
    # Social/Human (30-39)
    "trust",
    "power",
    "culture",
    "love",
    "fear",
    "ritual",
    "storytelling",
    "money",
    "hierarchy",
    "cooperation",
    
    # Concrete (40-49)
    "a library",
    "a mirror",
    "a bridge",
    "a seed",
    "a storm",
    "a clock",
    "a map",
    "a door",
    "a fire",
    "a river",
]

# Generate training pairs (random combinations from first 40 concepts)
# and eval pairs (combinations involving the last 10 concepts)
def generate_pairs():
    """Generate train and eval pairs."""
    train_concepts = CONCEPTS[:40]
    eval_concepts = CONCEPTS[40:]
    
    # Training pairs: random pairs from train concepts
    train_pairs = []
    for i in range(len(train_concepts)):
        for j in range(i + 1, len(train_concepts)):
            train_pairs.append((train_concepts[i], train_concepts[j]))
    random.shuffle(train_pairs)
    train_pairs = train_pairs[:200]  # Limit to 200 training pairs
    
    # Eval pairs: pairs that include at least one eval concept
    eval_pairs = []
    for ec in eval_concepts:
        for tc in train_concepts[:10]:  # Pair with first 10 train concepts
            eval_pairs.append((ec, tc))
    random.shuffle(eval_pairs)
    eval_pairs = eval_pairs[:20]  # 20 eval pairs
    
    return train_pairs, eval_pairs


TRAIN_PAIRS, EVAL_PAIRS = generate_pairs()


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
        "keys": [],
        "values": [],
        "num_tokens": 0,
    }


def get_kv_cache_from_text(model, tokenizer, text, num_tokens):
    """
    Run text through the model and extract the first num_tokens of the KV cache.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    if len(tokens) < num_tokens:
        tokens = tokens * ((num_tokens // len(tokens)) + 1)
    tokens = tokens[:num_tokens]
    
    input_ids = torch.tensor([tokens], device=DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
    
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
    """Freeze all current cartridge parameters."""
    for i in range(len(cartridge["keys"])):
        cartridge["keys"][i] = cartridge["keys"][i].detach().clone()
        cartridge["values"][i] = cartridge["values"][i].detach().clone()
    return cartridge


# =============================================================================
# GENERATION WITH CARTRIDGE
# =============================================================================

def generate_with_cartridge(model, tokenizer, model_info, concept_a, concept_b, 
                            skill_text, cartridge, max_tokens=300, temperature=0.8):
    """
    Generate a connection between two concepts using the model with cartridge + skill.
    """
    full_prompt = f"""You are a creative thinker who finds deep, non-obvious connections between ideas.

{skill_text}

Find a connection between: "{concept_a}" and "{concept_b}"

Write 2-4 sentences explaining a surprising, insightful connection:"""
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    cart_keys, cart_values = get_cartridge_tensors(cartridge)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
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


def generate_samples(model, tokenizer, model_info, concept_a, concept_b, 
                     skill_text, cartridge, n_samples=4):
    """Generate n samples for a concept pair."""
    samples = []
    for _ in range(n_samples):
        sample = generate_with_cartridge(
            model, tokenizer, model_info, concept_a, concept_b, skill_text, cartridge
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


def judge_sample(sample, concept_a, concept_b):
    """Have GPT-4.1 score a connection using the rubric."""
    judge_prompt = f"""You are judging the quality of a conceptual connection.

The task was to find a connection between: "{concept_a}" and "{concept_b}"

They wrote:
---
{sample}
---

{CONNECTION_RUBRIC}

Respond with ONLY the JSON scores, nothing else."""
    
    messages = [{"role": "user", "content": judge_prompt}]
    response = call_openai(messages, model="gpt-4.1", temperature=0.3)
    
    try:
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        scores = json.loads(response.strip())
    except json.JSONDecodeError:
        scores = {"novelty": 5, "coherence": 5, "depth": 5, "generativity": 5, "total": 5}
    
    return scores


def judge_samples(samples, concept_a, concept_b):
    """Judge multiple samples."""
    return [judge_sample(sample, concept_a, concept_b) for sample in samples]


def update_skill(current_skill, samples_with_scores, concept_a, concept_b, 
                 max_skill_tokens=512, tokenizer=None):
    """
    Have GPT-4.1 update the skill based on the samples and scores.
    """
    samples_text = ""
    for i, (sample, scores) in enumerate(samples_with_scores):
        samples_text += f"\n--- Sample {i+1} (total: {scores.get('total', 'N/A')}) ---\n"
        samples_text += f"{sample}\n"
        samples_text += f"Scores: {json.dumps(scores)}\n"
    
    update_prompt = f"""You are helping improve instructions for finding creative connections between concepts.

The current instructions are:
---
{current_skill}
---

The task was to connect: "{concept_a}" and "{concept_b}"

Here are the outputs and scores (1-10 scale, higher is better):
{samples_text}

Based on these results, please EDIT the instructions to help find BETTER connections.

IMPORTANT CONSTRAINTS:
- Keep instructions CONCISE - maximum ~{max_skill_tokens} tokens
- Focus on the MOST IMPORTANT strategies only
- Be specific: what makes a connection novel? deep? generative?
- Remove redundant or low-impact instructions

Output ONLY the new instructions, nothing else:"""

    messages = [{"role": "user", "content": update_prompt}]
    new_skill = call_openai(messages, model="gpt-4.1", temperature=0.7)
    new_skill = new_skill.strip()
    
    if tokenizer is not None:
        tokens = tokenizer.encode(new_skill, add_special_tokens=False)
        if len(tokens) > max_skill_tokens:
            tokens = tokens[:max_skill_tokens]
            new_skill = tokenizer.decode(tokens)
            last_period = new_skill.rfind('.')
            if last_period > len(new_skill) // 2:
                new_skill = new_skill[:last_period + 1]
    
    return new_skill


def generate_baseline_sample_self(model, tokenizer, model_info, concept_a, concept_b):
    """Generate a sample using the same model but WITHOUT skill or cartridge."""
    empty_cartridge = create_empty_cartridge(model_info)
    no_skill = "Find interesting connections between concepts."
    
    return generate_with_cartridge(
        model, tokenizer, model_info, concept_a, concept_b, no_skill, empty_cartridge,
        max_tokens=300, temperature=0.8
    )


def generate_baseline_sample_external(concept_a, concept_b, baseline_model="gpt-4.1"):
    """Generate a connection using an external API model."""
    gen_prompt = f"""Find a surprising, insightful connection between: "{concept_a}" and "{concept_b}"

Write 2-4 sentences explaining a non-obvious, deep connection between these concepts. 
Look for structural similarities, shared patterns, or unexpected relationships."""

    messages = [{"role": "user", "content": gen_prompt}]
    return call_openai(messages, model=baseline_model, temperature=0.8, max_tokens=400)


def generate_baseline_samples(concept_a, concept_b, n_samples=2, baseline_mode="self", 
                               baseline_model="gpt-4.1", model=None, tokenizer=None, model_info=None):
    """Generate n baseline samples."""
    samples = []
    for _ in range(n_samples):
        if baseline_mode == "self":
            sample = generate_baseline_sample_self(model, tokenizer, model_info, concept_a, concept_b)
        else:
            sample = generate_baseline_sample_external(concept_a, concept_b, baseline_model)
        samples.append(sample)
    return samples


# =============================================================================
# EVALUATION
# =============================================================================

def compare_pair(sample_a, sample_b, concept_a, concept_b):
    """Have GPT-4.1 judge which connection is better. Returns 'A', 'B', or 'TIE'."""
    compare_prompt = f"""You are judging which text provides a BETTER conceptual connection.
The concepts to connect were: "{concept_a}" and "{concept_b}"

Connection A:
---
{sample_a}
---

Connection B:
---
{sample_b}
---

Which connection is BETTER? Consider:
- NOVELTY: Is it non-obvious and surprising?
- COHERENCE: Does it actually make sense?
- DEPTH: Is it a deep insight or just surface wordplay?
- GENERATIVITY: Does it spark further thinking?

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


def direct_comparison_eval(our_samples, baseline_samples, concept_a, concept_b):
    """Direct 1:1 comparison."""
    our_wins = 0
    total = 0
    matchups = []
    
    n = min(len(our_samples), len(baseline_samples))
    
    for i in range(n):
        result = compare_pair(our_samples[i], baseline_samples[i], concept_a, concept_b)
        
        matchups.append({"idx": i, "result": result})
        
        total += 1
        if result == "A":
            our_wins += 1
        elif result == "TIE":
            our_wins += 0.5
    
    return our_wins, total, matchups


def run_full_eval(model, tokenizer, model_info, skill_text, cartridge, eval_pairs,
                  n_samples=2, baseline_mode="self", baseline_model="gpt-4.1", num_eval_pairs=10):
    """Run evaluation on held-out concept pairs."""
    results = {
        "pairs": [],
        "total_wins": 0,
        "total_matchups": 0,
        "baseline_mode": baseline_mode,
        "baseline_model": baseline_model if baseline_mode == "external" else "self (no skill/cartridge)",
    }
    
    pairs_to_eval = eval_pairs[:num_eval_pairs]
    
    for concept_a, concept_b in pairs_to_eval:
        print(f"    Evaluating: {concept_a} ↔ {concept_b}...")
        
        our_samples = generate_samples(
            model, tokenizer, model_info, concept_a, concept_b, skill_text, cartridge, n_samples=n_samples
        )
        
        baseline_samples = generate_baseline_samples(
            concept_a, concept_b, n_samples=n_samples, baseline_mode=baseline_mode, 
            baseline_model=baseline_model, model=model, tokenizer=tokenizer, model_info=model_info
        )
        
        wins, total, matchups = direct_comparison_eval(our_samples, baseline_samples, concept_a, concept_b)
        
        results["pairs"].append({
            "concept_a": concept_a,
            "concept_b": concept_b,
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
                         frozen_cartridge, skill_text, concept_a, concept_b, optimizer):
    """One step of cartridge training via context distillation."""
    question = f"Find a connection between: \"{concept_a}\" and \"{concept_b}\"\n\nWrite 2-4 sentences:"
    
    teacher_messages = [
        {"role": "system", "content": f"Follow these instructions:\n{skill_text}"},
        {"role": "user", "content": question}
    ]
    teacher_text = tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
    
    teacher_input_ids = tokenizer(teacher_text, return_tensors="pt").input_ids.to(DEVICE)
    
    with torch.no_grad():
        teacher_output = model.generate(
            teacher_input_ids,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer_ids = teacher_output[0, teacher_input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    teacher_full_ids = teacher_output
    teacher_prompt_len = teacher_input_ids.shape[1]
    
    with torch.no_grad():
        teacher_out = model(input_ids=teacher_full_ids, use_cache=False)
        teacher_logits = teacher_out.logits[0, teacher_prompt_len-1:-1, :]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        top_k = 20
        topk_probs, topk_ids = torch.topk(teacher_probs, k=top_k, dim=-1)
    
    student_messages = [{"role": "user", "content": question}]
    student_text = tokenizer.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True)
    student_full_text = student_text + answer_text
    student_tokens = tokenizer(student_full_text, return_tensors="pt").input_ids.to(DEVICE)
    student_prompt_len = len(tokenizer(student_text).input_ids)
    
    frozen_keys, frozen_values = get_cartridge_tensors(frozen_cartridge)
    
    if frozen_keys is not None:
        full_keys = torch.cat([frozen_keys, trainable_keys], dim=2)
        full_values = torch.cat([frozen_values, trainable_values], dim=2)
    else:
        full_keys = trainable_keys
        full_values = trainable_values
    
    cart_len = full_keys.shape[2]
    
    cache = DynamicCache()
    for layer_idx in range(model_info["num_layers"]):
        cache.update(
            full_keys[layer_idx].unsqueeze(0),
            full_values[layer_idx].unsqueeze(0),
            layer_idx,
        )
    
    attn_mask = torch.ones(1, cart_len + student_tokens.shape[1], device=DEVICE)
    position_ids = torch.arange(student_tokens.shape[1], device=DEVICE).unsqueeze(0) + cart_len
    
    student_out = model(
        input_ids=student_tokens,
        attention_mask=attn_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    
    student_logits = student_out.logits[0, student_prompt_len-1:-1, :]
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    num_tokens = min(student_logits.shape[0], topk_probs.shape[0])
    
    student_topk_logprobs = student_log_probs[:num_tokens].gather(
        dim=-1, index=topk_ids[:num_tokens]
    )
    
    ce_by_token = -(topk_probs[:num_tokens] * student_topk_logprobs).sum(dim=-1)
    loss = ce_by_token.mean()
    
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
    parser = argparse.ArgumentParser(description="ENGRAM: Creative Concept Linking (Daydreaming)")
    parser.add_argument("--output", type=str, default="daydream_run", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--iterations", type=int, default=100, help="Number of outer loop iterations")
    parser.add_argument("--skill-rounds", type=int, default=10, help="Skill refinement rounds per iteration")
    parser.add_argument("--cartridge-steps", type=int, default=50, help="Cartridge training steps per iteration")
    parser.add_argument("--tokens-per-iter", type=int, default=32, help="New cartridge tokens per iteration")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate for cartridge training")
    parser.add_argument("--samples-per-prompt", type=int, default=2, help="Samples to generate per prompt")
    parser.add_argument("--max-skill-tokens", type=int, default=512, help="Max tokens for skill file")
    # Eval settings
    parser.add_argument("--baseline-mode", type=str, default="self", choices=["self", "external"],
                        help="Baseline: 'self' = no skill/cartridge, 'external' = API model")
    parser.add_argument("--baseline-model", type=str, default="gpt-4.1", 
                        help="External baseline model (only used if --baseline-mode=external)")
    parser.add_argument("--eval-every", type=int, default=1, help="Run eval every N iterations")
    parser.add_argument("--num-eval-pairs", type=int, default=5, help="Number of eval pairs to use")
    parser.add_argument("--eval-samples", type=int, default=2, help="Samples per pair for eval")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    setup_logging(output_dir)
    
    print("=" * 70)
    print("ENGRAM: CREATIVE CONCEPT LINKING (DAYDREAMING LOOP)")
    print("Text Skill Refinement + Cartridge Condensation")
    print("=" * 70)
    
    save_json(vars(args), output_dir / "config.json")
    
    model, tokenizer, model_info = load_model(args.model)
    
    for param in model.parameters():
        param.requires_grad = False
    
    current_skill = INITIAL_SKILL
    cartridge = create_empty_cartridge(model_info)
    
    save_text(current_skill, output_dir / "skills" / "skill_iter_0.md")
    
    metrics_history = []
    
    print(f"\nStarting training for {args.iterations} iterations")
    print(f"  Skill rounds per iteration: {args.skill_rounds}")
    print(f"  Max skill tokens: {args.max_skill_tokens}")
    print(f"  Cartridge steps per iteration: {args.cartridge_steps}")
    print(f"  New cartridge tokens per iteration: {args.tokens_per_iter}")
    print(f"  Training pairs: {len(TRAIN_PAIRS)}")
    print(f"  Eval pairs: {len(EVAL_PAIRS)}")
    print(f"  Baseline mode: {args.baseline_mode}")
    print()
    
    # Initial eval
    print(f"\n[INITIAL EVAL] Before training starts...")
    initial_eval = run_full_eval(
        model, tokenizer, model_info, current_skill, cartridge, EVAL_PAIRS,
        n_samples=args.eval_samples, baseline_mode=args.baseline_mode,
        baseline_model=args.baseline_model, num_eval_pairs=args.num_eval_pairs
    )
    save_json(initial_eval, output_dir / "eval_logs" / "initial_eval.json")
    print(f"  Initial win rate: {initial_eval['overall_win_rate']:.1%}")
    
    metrics_history.append({
        "iteration": 0,
        "cartridge_tokens": 0,
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
        
        # Reset skill each iteration
        current_skill = INITIAL_SKILL
        
        do_eval = (iteration % args.eval_every == 0)
        
        # =====================================================================
        # PHASE A: Skill Refinement
        # =====================================================================
        print(f"\n[PHASE A] Skill Refinement ({args.skill_rounds} rounds)")
        
        for round_idx in range(args.skill_rounds):
            concept_a, concept_b = random.choice(TRAIN_PAIRS)
            
            print(f"  Round {round_idx + 1}: {concept_a} ↔ {concept_b}...")
            
            samples = generate_samples(
                model, tokenizer, model_info, concept_a, concept_b, current_skill, cartridge,
                n_samples=args.samples_per_prompt
            )
            
            scores = judge_samples(samples, concept_a, concept_b)
            
            samples_with_scores = list(zip(samples, scores))
            avg_score = sum(s.get("total", 5) for s in scores) / len(scores)
            
            old_skill = current_skill
            current_skill = update_skill(current_skill, samples_with_scores, concept_a, concept_b,
                                         max_skill_tokens=args.max_skill_tokens, tokenizer=tokenizer)
            
            round_log = {
                "round": round_idx + 1,
                "concept_a": concept_a,
                "concept_b": concept_b,
                "samples": samples,
                "scores": scores,
                "avg_score": avg_score,
                "old_skill": old_skill,
                "new_skill": current_skill,
            }
            iter_log["skill_rounds"].append(round_log)
            
            print(f"    Avg score: {avg_score:.1f}/10")
        
        save_text(current_skill, output_dir / "skills" / f"skill_iter_{iteration}_after_phase_a.md")
        
        # =====================================================================
        # EVAL: After Phase A
        # =====================================================================
        if do_eval:
            print("\n[EVAL] After skill refinement...")
            eval_after_skill = run_full_eval(
                model, tokenizer, model_info, current_skill, cartridge, EVAL_PAIRS,
                n_samples=args.eval_samples, baseline_mode=args.baseline_mode,
                baseline_model=args.baseline_model, num_eval_pairs=args.num_eval_pairs
            )
            iter_log["evals"]["after_skill"] = eval_after_skill
            print(f"  Win rate: {eval_after_skill['overall_win_rate']:.1%}")
        else:
            eval_after_skill = None
        
        # =====================================================================
        # PHASE B: Cartridge Training
        # =====================================================================
        print(f"\n[PHASE B] Cartridge Training ({args.cartridge_steps} steps)")
        
        cartridge = freeze_cartridge(cartridge)
        
        new_keys, new_values = get_kv_cache_from_text(
            model, tokenizer, current_skill, args.tokens_per_iter
        )
        
        trainable_keys = torch.nn.Parameter(new_keys.clone())
        trainable_values = torch.nn.Parameter(new_values.clone())
        
        optimizer = torch.optim.Adam([trainable_keys, trainable_values], lr=args.lr)
        
        losses = []
        for step in range(args.cartridge_steps):
            concept_a, concept_b = random.choice(TRAIN_PAIRS)
            
            loss, answer = train_cartridge_step(
                model, tokenizer, model_info,
                trainable_keys, trainable_values,
                cartridge, current_skill, concept_a, concept_b, optimizer
            )
            
            losses.append(loss)
            
            iter_log["cartridge_training"].append({
                "step": step + 1,
                "concept_a": concept_a,
                "concept_b": concept_b,
                "loss": loss,
            })
            
            if (step + 1) % 10 == 0:
                avg_loss = sum(losses[-10:]) / 10
                print(f"  Step {step + 1}: loss = {avg_loss:.4f}")
        
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
                model, tokenizer, model_info, current_skill, cartridge, EVAL_PAIRS,
                n_samples=args.eval_samples, baseline_mode=args.baseline_mode,
                baseline_model=args.baseline_model, num_eval_pairs=args.num_eval_pairs
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
        
        torch.save({
            "keys": cartridge["keys"],
            "values": cartridge["values"],
            "num_tokens": cartridge["num_tokens"],
        }, output_dir / "cartridge_logs" / f"cartridge_iter_{iteration:04d}.pt")
        
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
    
    print("\nWin rate progression (after cartridge):")
    for m in metrics_history[::max(1, len(metrics_history)//10)]:
        if m.get('win_rate_after_cartridge') is not None:
            print(f"  Iter {m['iteration']:3d}: {m['win_rate_after_cartridge']:.1%}")


if __name__ == "__main__":
    main()

