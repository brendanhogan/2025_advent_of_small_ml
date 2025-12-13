"""
simple_train.py - Train a cartridge on one QuALITY story

Run: python simple_train.py --output my_run --steps 200 --tokens 512
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from tqdm import tqdm
import random
import json
from pathlib import Path
import inspect
import math

# Optional OpenAI import for synthetic eval generation
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Top-k for storing teacher logprobs during synthesis
TOP_K = 20


def load_model(model_name):
    """Load model and tokenizer."""
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    print(f"Model: {num_layers} layers, {num_kv_heads} KV heads, {head_dim} head dim")
    
    return model, tokenizer, num_layers, num_kv_heads, head_dim


def load_quality_dataset(story_idx=0):
    """Load one story from QuALITY dataset."""
    print(f"\nLoading QuALITY dataset (story {story_idx})...")
    dataset = load_dataset("emozilla/quality", split="train")
    
    # Get unique articles
    unique_articles = []
    seen = set()
    for row in dataset:
        if row["article"] not in seen:
            unique_articles.append(row["article"])
            seen.add(row["article"])
    
    if story_idx >= len(unique_articles):
        raise ValueError(f"story_idx {story_idx} out of range (only {len(unique_articles)} stories)")
    
    article = unique_articles[story_idx]
    eval_questions = []
    
    for row in dataset:
        if row["article"] == article:
            eval_questions.append({
                "question": row["question"],
                "options": row["options"],
                "answer_idx": row["answer"],
            })
    
    print(f"Story {story_idx}: {len(article)} chars, {len(eval_questions)} questions")
    return article, eval_questions, story_idx


def create_cartridge_random(num_layers, num_kv_heads, num_tokens, head_dim):
    """Create random cartridge parameters (old method, for comparison)."""
    print(f"\nCreating RANDOM cartridge: {num_tokens} tokens")
    
    keys = nn.Parameter(torch.randn(
        num_layers, num_kv_heads, num_tokens, head_dim,
        dtype=DTYPE, device=DEVICE
    ) * 0.02)
    
    values = nn.Parameter(torch.randn(
        num_layers, num_kv_heads, num_tokens, head_dim,
        dtype=DTYPE, device=DEVICE
    ) * 0.02)
    
    print(f"Cartridge params: {keys.numel() + values.numel():,}")
    return keys, values


def create_cartridge_from_text(model, tokenizer, num_layers, num_kv_heads, num_tokens, head_dim, text):
    """
    Initialize cartridge by running text through the model and extracting the KV cache.
    This puts the cartridge in a "reasonable" region of KV space rather than random noise.
    """
    print(f"\nCreating TEXT-INITIALIZED cartridge: {num_tokens} tokens")
    
    # Tokenize text, truncate to num_tokens
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > num_tokens:
        tokens = tokens[:num_tokens]
    elif len(tokens) < num_tokens:
        # Pad by repeating if needed
        repeats = (num_tokens // len(tokens)) + 1
        tokens = (tokens * repeats)[:num_tokens]
    
    input_ids = torch.tensor([tokens], device=DEVICE)
    
    # Run through model to get KV cache
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_kv = outputs.past_key_values
    
    # Extract keys and values from the cache
    # past_kv is a tuple of (key, value) for each layer
    # Each key/value has shape (batch, num_kv_heads, seq_len, head_dim)
    keys_list = []
    values_list = []
    for layer_idx in range(num_layers):
        k, v = past_kv[layer_idx]
        keys_list.append(k[0])  # Remove batch dim
        values_list.append(v[0])
    
    keys = nn.Parameter(torch.stack(keys_list, dim=0).to(DTYPE))
    values = nn.Parameter(torch.stack(values_list, dim=0).to(DTYPE))
    
    print(f"Cartridge initialized from {len(tokens)} tokens of text")
    print(f"Cartridge params: {keys.numel() + values.numel():,}")
    return keys, values


def _model_accepts_arg(model, arg_name: str) -> bool:
    try:
        sig = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return arg_name in sig.parameters


def _build_cartridge_cache_and_masks(model, input_ids, num_layers, cartridge_keys, cartridge_values):
    """
    Build a DynamicCache prefilled with cartridge KV, and construct attention_mask + position_ids
    so that the input tokens are positioned *after* the cartridge tokens.
    """
    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(
            cartridge_keys[layer_idx].unsqueeze(0),
            cartridge_values[layer_idx].unsqueeze(0),
            layer_idx,
        )

    cart_len = cartridge_keys.shape[2]
    seq_len = input_ids.shape[1]
    attn_mask = torch.ones(1, cart_len + seq_len, device=input_ids.device)

    # Offset RoPE / position embeddings by cartridge length.
    position_ids = (torch.arange(seq_len, device=input_ids.device).unsqueeze(0) + cart_len)

    # Some HF models also accept `cache_position`, which influences cache-related indexing.
    cache_position = None
    if _model_accepts_arg(model, "cache_position"):
        cache_position = position_ids[0].clone()

    return cache, attn_mask, position_ids, cache_position


def generate(model, tokenizer, num_layers, prompt, cartridge=None, context=None, max_tokens=50, temperature=0.7):
    """Generate text, optionally with cartridge or context."""
    if context:
        full_prompt = f"Context:\n{context}\n\n{prompt}"
    else:
        full_prompt = prompt
    
    messages = [{"role": "user", "content": full_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            if cartridge is not None:
                keys, values = cartridge
                cache, attn_mask, position_ids, cache_position = _build_cartridge_cache_and_masks(
                    model=model,
                    input_ids=input_ids,
                    num_layers=num_layers,
                    cartridge_keys=keys,
                    cartridge_values=values,
                )
            else:
                cache = None
                attn_mask = torch.ones(1, input_ids.shape[1], device=DEVICE)
                position_ids = None
                cache_position = None
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=(cache is not None),
            )
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated_ids.append(next_token.item())
            if next_token.item() == tokenizer.eos_token_id:
                break
            input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate(name, model, tokenizer, num_layers, eval_questions, cartridge=None, context=None, num_samples=3, max_questions=10):
    """Evaluate and return detailed results."""
    questions = eval_questions[:max_questions]
    
    results = {
        "name": name,
        "num_samples": num_samples,
        "num_questions": len(questions),
        "questions": [],
    }
    
    correct = 0
    pbar = tqdm(questions, desc=f"Eval {name}", leave=False)
    for q_idx, q in enumerate(pbar):
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(q["options"])])
        prompt = f"""Question: {q["question"]}

{options_str}

Answer with just the letter (A, B, C, or D):"""
        
        q_result = {
            "question": q["question"],
            "options": q["options"],
            "ground_truth_idx": q["answer_idx"],
            "ground_truth_letter": chr(65 + q["answer_idx"]),
            "samples": [],
            "any_correct": False,
        }
        
        for _ in range(num_samples):
            response = generate(model, tokenizer, num_layers, prompt, cartridge=cartridge, context=context, max_tokens=10, temperature=0.7)
            response_clean = response.strip().upper()
            
            predicted_idx = None
            predicted_letter = None
            for i, letter in enumerate(['A', 'B', 'C', 'D']):
                if letter in response_clean[:5]:
                    predicted_idx = i
                    predicted_letter = letter
                    break
            
            is_correct = predicted_idx == q["answer_idx"]
            if is_correct:
                q_result["any_correct"] = True
            
            q_result["samples"].append({
                "raw_response": response,
                "predicted_letter": predicted_letter,
                "correct": is_correct,
            })
        
        if q_result["any_correct"]:
            correct += 1
        
        results["questions"].append(q_result)
        pbar.set_postfix({"acc": f"{correct}/{q_idx+1}"})
    
    results["accuracy"] = correct / len(questions)
    results["correct"] = correct
    results["total"] = len(questions)
    
    print(f"  Pass@{num_samples}: {results['accuracy']:.1%} ({correct}/{len(questions)})")
    return results


def generate_synthetic_qa_with_topk(model, tokenizer, num_layers, article, chunk_size=2000, questions_per_chunk=15, top_k=TOP_K):
    """
    Generate synthetic Q&A pairs from article chunks.
    Also store top-k teacher logprobs for each answer token (for sparse CE training).
    """
    chunks = [article[i:i+chunk_size] for i in range(0, len(article), chunk_size)]
    synthetic_qa = []
    
    total_questions = len(chunks) * questions_per_chunk
    print(f"\nGenerating ~{total_questions} synthetic Q&A from {len(chunks)} chunks (with top-{top_k} logprobs)...")
    
    question_types = [
        "Generate a specific factual question about a detail in this text.",
        "Generate a question about who, what, or where something happened in this text.",
        "Generate a question that asks about a number, date, or quantity in this text.",
        "Generate a question about the sequence of events in this text.",
        "Generate a question about a person or character mentioned in this text.",
    ]
    
    for chunk_idx, chunk in enumerate(chunks):
        for q_num in range(questions_per_chunk):
            q_type = question_types[q_num % len(question_types)]
            q_prompt = f"""Based on this text, {q_type}

Text:
{chunk}

Generate only the question:"""
            
            question = generate(model, tokenizer, num_layers, q_prompt, max_tokens=50, temperature=0.9)
            
            a_prompt = f"Question: {question}\n\nAnswer briefly:"
            answer = generate(model, tokenizer, num_layers, a_prompt, context=chunk, max_tokens=100, temperature=0.3)
            
            # Now get teacher's top-k logprobs for this answer
            teacher_messages = [{"role": "user", "content": f"Context:\n{chunk}\n\n{a_prompt}"}]
            teacher_text = tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
            teacher_full = teacher_text + answer
            teacher_tokens = tokenizer(teacher_full, return_tensors="pt").input_ids.to(DEVICE)
            teacher_prompt_len = len(tokenizer(teacher_text).input_ids)
            
            with torch.no_grad():
                teacher_out = model(input_ids=teacher_tokens, use_cache=False)
                # Logits for answer positions: predict tokens [prompt_len:] using logits [prompt_len-1:-1]
                answer_logits = teacher_out.logits[0, teacher_prompt_len-1:-1, :]  # [num_answer_tokens, vocab]
                answer_probs = F.softmax(answer_logits, dim=-1)
                
                # Get top-k for each position
                topk_probs, topk_ids = torch.topk(answer_probs, k=top_k, dim=-1)  # [num_answer_tokens, top_k]
                
                # Store as lists (JSON serializable)
                topk_probs_list = topk_probs.cpu().tolist()
                topk_ids_list = topk_ids.cpu().tolist()
            
            synthetic_qa.append({
                "chunk_idx": chunk_idx,
                "question": question,
                "answer": answer,
                "topk_probs": topk_probs_list,  # [num_answer_tokens, top_k]
                "topk_ids": topk_ids_list,      # [num_answer_tokens, top_k]
            })
        
        print(f"  Chunk {chunk_idx+1}/{len(chunks)}: {questions_per_chunk} Q&A generated")
    
    print(f"\nTotal: {len(synthetic_qa)} synthetic Q&A pairs with top-{top_k} logprobs")
    return synthetic_qa, chunks


def generate_synthetic_mc_eval_openai(article, num_questions=50, chunk_size=3000, model_name="gpt-4.1"):
    """
    Generate synthetic multiple-choice eval questions using GPT-4.1-mini.
    Higher quality than local model generation.
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not installed. Run: pip install openai")
    
    client = OpenAI()  # Uses OPENAI_API_KEY from environment
    
    chunks = [article[i:i+chunk_size] for i in range(0, len(article), chunk_size)]
    mc_questions = []
    
    questions_per_chunk = max(1, (num_questions + len(chunks) - 1) // len(chunks))
    print(f"\nGenerating {num_questions} synthetic MC eval questions using {model_name}...")
    print(f"  {len(chunks)} chunks, ~{questions_per_chunk} questions per chunk")
    
    for chunk_idx, chunk in enumerate(tqdm(chunks, desc="Generating MC questions")):
        remaining = num_questions - len(mc_questions)
        if remaining <= 0:
            break
        
        n_for_chunk = min(questions_per_chunk, remaining)
        
        prompt = f"""Based on the following text passage, generate exactly {n_for_chunk} multiple-choice questions.

TEXT PASSAGE:
{chunk}

INSTRUCTIONS:
1. Each question should test comprehension of specific facts from the text
2. Questions should have clear, unambiguous answers based on the text
3. Each question needs exactly 4 options (A, B, C, D)
4. One option must be correct, three must be plausible but wrong
5. Vary question types: who, what, where, when, why, how many, etc.

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "What is the question?",
    "options": ["correct answer", "wrong answer 1", "wrong answer 2", "wrong answer 3"],
    "correct_idx": 0
  }},
  ...
]

Generate exactly {n_for_chunk} questions. Output ONLY the JSON array, no other text."""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse JSON - handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            questions_data = json.loads(content)
            
            for q_data in questions_data:
                # Shuffle options and track correct index
                correct_answer = q_data["options"][q_data["correct_idx"]]
                options = q_data["options"][:]
                
                indices = list(range(4))
                random.shuffle(indices)
                shuffled_options = [options[i] for i in indices]
                new_correct_idx = indices.index(q_data["correct_idx"])
                
                mc_questions.append({
                    "question": q_data["question"],
                    "options": shuffled_options,
                    "answer_idx": new_correct_idx,
                    "chunk_idx": chunk_idx,
                    "source": "synthetic_gpt4",
                })
                
        except json.JSONDecodeError as e:
            print(f"  Warning: Failed to parse JSON for chunk {chunk_idx}: {e}")
            continue
        except Exception as e:
            print(f"  Warning: API error for chunk {chunk_idx}: {e}")
            continue
    
    print(f"\nTotal: {len(mc_questions)} synthetic MC eval questions generated")
    return mc_questions


def generate_synthetic_mc_eval_local(model, tokenizer, num_layers, article, num_questions=50, chunk_size=2000):
    """
    Generate synthetic multiple-choice eval questions using the local model.
    Fallback if OpenAI API not available.
    """
    chunks = [article[i:i+chunk_size] for i in range(0, len(article), chunk_size)]
    mc_questions = []
    
    questions_per_chunk = max(1, num_questions // len(chunks))
    print(f"\nGenerating {num_questions} synthetic MC eval questions from {len(chunks)} chunks (local model)...")
    
    question_types = [
        "Generate a specific factual question about a detail in this text that has a clear, short answer.",
        "Generate a question about who did something or who is mentioned in this text.",
        "Generate a question about what happened or what something is in this text.",
        "Generate a question about where or when something happened in this text.",
        "Generate a question about a number, amount, or quantity mentioned in this text.",
    ]
    
    generated = 0
    for chunk_idx, chunk in enumerate(chunks):
        for q_num in range(questions_per_chunk):
            if generated >= num_questions:
                break
                
            q_type = question_types[q_num % len(question_types)]
            
            # Generate question
            q_prompt = f"""Based on this text, {q_type}

Text:
{chunk}

Generate only the question (make sure it has a definite answer from the text):"""
            
            question = generate(model, tokenizer, num_layers, q_prompt, max_tokens=50, temperature=0.8)
            
            # Get correct answer with context
            a_prompt = f"Question: {question}\n\nAnswer in 1-5 words:"
            correct_answer = generate(model, tokenizer, num_layers, a_prompt, context=chunk, max_tokens=20, temperature=0.3)
            correct_answer = correct_answer.strip()
            
            # Generate distractors (plausible but wrong answers)
            distractor_prompt = f"""Question: {question}
Correct answer: {correct_answer}

Generate 3 plausible but WRONG answers to this question. They should sound reasonable but be incorrect based on the text. Format as:
1. [wrong answer 1]
2. [wrong answer 2]  
3. [wrong answer 3]"""
            
            distractors_raw = generate(model, tokenizer, num_layers, distractor_prompt, context=chunk, max_tokens=100, temperature=0.7)
            
            # Parse distractors
            distractors = []
            for line in distractors_raw.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove leading number/bullet
                    answer = line.lstrip('0123456789.-) ').strip()
                    if answer and answer.lower() != correct_answer.lower():
                        distractors.append(answer)
            
            # Ensure we have exactly 3 distractors
            while len(distractors) < 3:
                distractors.append(f"None of the above ({len(distractors)+1})")
            distractors = distractors[:3]
            
            # Shuffle options and track correct index
            options = [correct_answer] + distractors
            correct_idx = 0
            
            # Shuffle
            indices = list(range(4))
            random.shuffle(indices)
            options = [options[i] for i in indices]
            correct_idx = indices.index(0)
            
            mc_questions.append({
                "question": question,
                "options": options,
                "answer_idx": correct_idx,
                "chunk_idx": chunk_idx,
                "source": "synthetic_local",
            })
            generated += 1
        
        if generated >= num_questions:
            break
        print(f"  Chunk {chunk_idx+1}/{len(chunks)}: generated")
    
    print(f"\nTotal: {len(mc_questions)} synthetic MC eval questions")
    return mc_questions


def get_cosine_schedule_lr(step, total_steps, base_lr, warmup_steps=100, min_lr_ratio=0.1):
    """Cosine schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model, tokenizer, num_layers, cartridge_keys, cartridge_values, synthetic_qa, chunks, 
          num_tokens, num_steps=100, lr=1e-3, eval_every=25, eval_questions=None, num_samples=3, 
          max_questions=10, out_dir=None, use_cosine_schedule=True, top_k=TOP_K):
    """Train cartridge via sparse top-k cross-entropy distillation."""
    optimizer = torch.optim.Adam([cartridge_keys, cartridge_values], lr=lr)
    
    for param in model.parameters():
        param.requires_grad = False
    
    training_log = []
    
    print(f"\nTraining for {num_steps} steps (top-{top_k} CE loss, cosine LR: {use_cosine_schedule})...")
    for step in range(num_steps):
        # Update LR if using cosine schedule
        if use_cosine_schedule:
            current_lr = get_cosine_schedule_lr(step, num_steps, lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = lr
        
        qa = random.choice(synthetic_qa)
        
        # Build student prompt
        question_prompt = f"Question: {qa['question']}\n\nAnswer briefly:"
        student_messages = [{"role": "user", "content": question_prompt}]
        student_text = tokenizer.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True)
        student_full = student_text + qa['answer']
        student_tokens = tokenizer(student_full, return_tensors="pt").input_ids.to(DEVICE)
        student_prompt_len = len(tokenizer(student_text).input_ids)
        
        # Student forward (with cartridge, no context)
        cache, attn_mask, position_ids, cache_position = _build_cartridge_cache_and_masks(
            model=model,
            input_ids=student_tokens,
            num_layers=num_layers,
            cartridge_keys=cartridge_keys,
            cartridge_values=cartridge_values,
        )
        student_out = model(
            input_ids=student_tokens,
            attention_mask=attn_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        # Logits for answer positions
        student_answer_logits = student_out.logits[0, student_prompt_len-1:-1, :]  # [num_answer_tokens, vocab]
        student_log_probs = F.log_softmax(student_answer_logits, dim=-1)
        
        # Load precomputed teacher top-k
        topk_probs = torch.tensor(qa['topk_probs'], device=DEVICE, dtype=DTYPE)  # [num_answer_tokens, top_k]
        topk_ids = torch.tensor(qa['topk_ids'], device=DEVICE, dtype=torch.long)  # [num_answer_tokens, top_k]
        
        num_answer_tokens = min(student_answer_logits.shape[0], topk_probs.shape[0])
        
        # Sparse top-k cross-entropy: -sum_k p_teacher(k) * log p_student(k)
        # Gather student log probs at teacher's top-k positions
        student_topk_logprobs = student_log_probs[:num_answer_tokens].gather(
            dim=-1, index=topk_ids[:num_answer_tokens]
        )  # [num_answer_tokens, top_k]
        
        # CE = -sum p(x) log q(x)
        ce_by_token = -(topk_probs[:num_answer_tokens] * student_topk_logprobs).sum(dim=-1)  # [num_answer_tokens]
        loss = ce_by_token.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_log.append({
            "step": step + 1, 
            "loss": loss.item(), 
            "answer_tokens": num_answer_tokens,
            "lr": current_lr,
        })
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{num_steps}, CE Loss: {loss.item():.4f} ({num_answer_tokens} tokens, lr={current_lr:.2e})")
        
        if (step + 1) % eval_every == 0 and eval_questions is not None:
            print(f"\n--- Eval at step {step+1} ---")
            results = evaluate(f"step_{step+1}", model, tokenizer, num_layers, eval_questions,
                             cartridge=(cartridge_keys, cartridge_values), num_samples=num_samples, max_questions=max_questions)
            if out_dir:
                with open(out_dir / f"eval_step_{step+1}.json", "w") as f:
                    json.dump(results, f, indent=2)
            print()
    
    return training_log


def main():
    parser = argparse.ArgumentParser(description="Train a cartridge on QuALITY")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--tokens", type=int, default=512, help="Number of cartridge tokens")
    parser.add_argument("--steps", type=int, default=2000, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=2e-2, help="Learning rate (paper uses 2e-2)")
    parser.add_argument("--eval-every", type=int, default=200, help="Eval every N steps")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples per question for pass@k")
    parser.add_argument("--max-questions", type=int, default=15, help="Max questions to eval")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size for synthetic QA")
    parser.add_argument("--questions-per-chunk", type=int, default=15, help="Synthetic Q&A per chunk")
    parser.add_argument("--story-idx", type=int, default=0, help="Which QuALITY story to use (0-indexed)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top-k for sparse CE loss")
    parser.add_argument("--init-mode", type=str, default="text", choices=["text", "random"],
                        help="Cartridge initialization: 'text' (from article) or 'random'")
    parser.add_argument("--no-cosine-schedule", action="store_true", help="Disable cosine LR schedule")
    parser.add_argument("--synthetic-eval", type=int, default=0, 
                        help="Number of synthetic MC eval questions to generate (0 = use only QuALITY questions)")
    parser.add_argument("--synthetic-eval-local", action="store_true",
                        help="Use local model for synthetic eval (default: use GPT-4.1-mini)")
    parser.add_argument("--synthetic-eval-model", type=str, default="gpt-4.1-mini",
                        help="OpenAI model for synthetic eval generation")
    args = parser.parse_args()
    
    # Setup output
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output: {out_dir}")
    
    # Load model and data
    model, tokenizer, num_layers, num_kv_heads, head_dim = load_model(args.model)
    article, quality_eval_questions, story_idx = load_quality_dataset(args.story_idx)
    
    # Count tokens in article
    article_tokens = len(tokenizer.encode(article))
    
    # Generate synthetic eval questions if requested
    if args.synthetic_eval > 0:
        print("\n" + "=" * 60)
        print(f"GENERATING {args.synthetic_eval} SYNTHETIC EVAL QUESTIONS")
        print("=" * 60)
        
        if args.synthetic_eval_local:
            synthetic_eval_questions = generate_synthetic_mc_eval_local(
                model, tokenizer, num_layers, article, 
                num_questions=args.synthetic_eval,
                chunk_size=args.chunk_size,
            )
        else:
            # Use GPT-4.1-mini for higher quality
            synthetic_eval_questions = generate_synthetic_mc_eval_openai(
                article, 
                num_questions=args.synthetic_eval,
                chunk_size=3000,  # Larger chunks for GPT-4
                model_name=args.synthetic_eval_model,
            )
        
        # Combine: synthetic first, then original QuALITY
        eval_questions = synthetic_eval_questions + quality_eval_questions
        print(f"Total eval questions: {len(eval_questions)} ({len(synthetic_eval_questions)} synthetic + {len(quality_eval_questions)} QuALITY)")
    else:
        eval_questions = quality_eval_questions
    
    # Build experiment info
    experiment = {
        "config": vars(args),
        "model": {
            "name": args.model,
            "num_layers": num_layers,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "cartridge": {
            "num_tokens": args.tokens,
            "num_params": 2 * num_layers * num_kv_heads * args.tokens * head_dim,
            "init_mode": args.init_mode,
        },
        "data": {
            "story_idx": story_idx,
            "article_chars": len(article),
            "article_tokens": article_tokens,
            "num_eval_questions": len(eval_questions),
            "num_quality_questions": len(quality_eval_questions),
            "num_synthetic_eval": args.synthetic_eval,
            "chunk_size": args.chunk_size,
        },
        "results": {},
    }
    
    print(f"\nArticle: {len(article)} chars, {article_tokens} tokens")
    print(f"Cartridge: {args.tokens} tokens, {experiment['cartridge']['num_params']:,} params")
    print(f"Init mode: {args.init_mode}")
    
    with open(out_dir / "article.txt", "w") as f:
        f.write(article)
    
    # Save eval questions (including synthetic if generated)
    if args.synthetic_eval > 0:
        with open(out_dir / "eval_questions.json", "w") as f:
            json.dump(eval_questions, f, indent=2)
    
    # Baseline: full context
    print("\n" + "=" * 60)
    print("BASELINE: Full context")
    print("=" * 60)
    context_results = evaluate("full_context", model, tokenizer, num_layers, eval_questions,
                               context=article, num_samples=args.num_samples, max_questions=args.max_questions)
    with open(out_dir / "eval_full_context.json", "w") as f:
        json.dump(context_results, f, indent=2)
    
    # Create cartridge
    if args.init_mode == "text":
        cartridge_keys, cartridge_values = create_cartridge_from_text(
            model, tokenizer, num_layers, num_kv_heads, args.tokens, head_dim, article
        )
    else:
        cartridge_keys, cartridge_values = create_cartridge_random(
            num_layers, num_kv_heads, args.tokens, head_dim
        )
    
    # Random/initial cartridge baseline
    print(f"\n{args.init_mode.capitalize()}-initialized cartridge (before training)...")
    init_results = evaluate("init_cartridge", model, tokenizer, num_layers, eval_questions,
                            cartridge=(cartridge_keys, cartridge_values), num_samples=args.num_samples, max_questions=args.max_questions)
    with open(out_dir / "eval_init_cartridge.json", "w") as f:
        json.dump(init_results, f, indent=2)
    
    # Self-study (with top-k logprobs)
    print("\n" + "=" * 60)
    print("SELF-STUDY (with top-k teacher logprobs)")
    print("=" * 60)
    synthetic_qa, chunks = generate_synthetic_qa_with_topk(
        model, tokenizer, num_layers, article, 
        chunk_size=args.chunk_size,
        questions_per_chunk=args.questions_per_chunk,
        top_k=args.top_k,
    )
    with open(out_dir / "synthetic_qa.json", "w") as f:
        # Save without the large topk arrays for readability
        synthetic_qa_small = [{k: v for k, v in qa.items() if k not in ['topk_probs', 'topk_ids']} for qa in synthetic_qa]
        json.dump(synthetic_qa_small, f, indent=2)
    # Save full version with topk data
    torch.save(synthetic_qa, out_dir / "synthetic_qa_full.pt")
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING (sparse top-k CE)")
    print("=" * 60)
    training_log = train(
        model, tokenizer, num_layers, cartridge_keys, cartridge_values, synthetic_qa, chunks,
        num_tokens=args.tokens, num_steps=args.steps, lr=args.lr, eval_every=args.eval_every,
        eval_questions=eval_questions, num_samples=args.num_samples, max_questions=args.max_questions,
        out_dir=out_dir, use_cosine_schedule=not args.no_cosine_schedule, top_k=args.top_k,
    )
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)
    
    # Final eval
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    print("\nFull context...")
    final_context = evaluate("final_context", model, tokenizer, num_layers, eval_questions,
                             context=article, num_samples=args.num_samples, max_questions=args.max_questions)
    with open(out_dir / "eval_final_context.json", "w") as f:
        json.dump(final_context, f, indent=2)
    
    print("\nTrained cartridge...")
    final_cartridge = evaluate("final_cartridge", model, tokenizer, num_layers, eval_questions,
                               cartridge=(cartridge_keys, cartridge_values), num_samples=args.num_samples, max_questions=args.max_questions)
    with open(out_dir / "eval_final_cartridge.json", "w") as f:
        json.dump(final_cartridge, f, indent=2)
    
    # Update experiment with results
    experiment["results"] = {
        "full_context_baseline": context_results["accuracy"],
        "init_cartridge": init_results["accuracy"],
        "final_cartridge": final_cartridge["accuracy"],
        "final_full_context": final_context["accuracy"],
    }
    experiment["data"]["num_chunks"] = len(chunks)
    experiment["training"] = {
        "num_steps": args.steps,
        "num_synthetic_qa": len(synthetic_qa),
        "final_loss": training_log[-1]["loss"] if training_log else None,
        "top_k": args.top_k,
        "cosine_schedule": not args.no_cosine_schedule,
    }
    
    with open(out_dir / "experiment.json", "w") as f:
        json.dump(experiment, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Article: {len(article)} chars, {article_tokens} tokens")
    print(f"  Cartridge: {args.tokens} tokens ({experiment['cartridge']['num_params']:,} params)")
    print(f"  Init mode: {args.init_mode}")
    print(f"  Full context:       {final_context['accuracy']:.1%}")
    print(f"  Trained cartridge:  {final_cartridge['accuracy']:.1%}")
    print(f"  Initial cartridge:  {init_results['accuracy']:.1%}")
    print("=" * 60)
    print(f"\nResults: {out_dir}/")


if __name__ == "__main__":
    main()
