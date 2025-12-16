"""
wiki_engram.py - ENGRAM for Wiki Search: Skill + Cartridge Learning

Combines:
- Wiki-search environment (tools, corpus, judge scoring via verifiers.JudgeRubric)
- ENGRAM-style continual learning (skill refinement + cartridge distillation)

The model learns to answer wiki trivia questions by:
1. Refining a skill.md with search strategies
2. Distilling the skill into a cartridge (KV cache vectors)
3. Repeating with the cartridge as accumulated memory

Uses the verifiers library for judging (JudgeRubric) and dataset.

Usage:
    python wiki_engram.py --output my_run --iterations 20
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import chromadb
import torch
import torch.nn.functional as F
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI, OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import verifiers as vf

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Initial skill (minimal)
INITIAL_SKILL = """You are answering trivia questions using Wikipedia search tools.

Available tools:
- search_pages(query): Find relevant Wikipedia pages by title
- view_sections(page_id): See the sections of a page  
- read_section(section_id): Read a specific section

Strategy: Search for relevant terms, browse sections, read to find the answer."""

# Judge prompt for scoring answers
JUDGE_PROMPT = """Given a ground truth answer and a response, determine if the response is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only.

If a response contains incoherent text, respond with "no" even if the correct answer is also present."""


# =============================================================================
# WIKI CORPUS + TOOLS
# =============================================================================

class WikiCorpus:
    """Manages the Wikipedia corpus and ChromaDB search."""
    
    def __init__(self, 
                 corpus_dataset="willcb/rare-wiki-pages",
                 questions_dataset="willcb/wiki-trivia-questions-v4",
                 chroma_db_dir=".chroma_db",
                 embed_model="text-embedding-3-small"):
        
        print("Loading Wikipedia corpus...")
        corpus = load_dataset(corpus_dataset, split="train")
        
        self.page_id_to_title = {}
        self.page_id_to_content = {}
        for row in corpus:
            pid = row["id"]
            self.page_id_to_title[pid] = row["title"]
            self.page_id_to_content[pid] = row["content"]
        
        print(f"  Loaded {len(self.page_id_to_title)} pages")
        
        # Setup ChromaDB
        print("Setting up ChromaDB...")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            model_name=embed_model,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
        client = chromadb.PersistentClient(path=chroma_db_dir)
        self.collection = client.get_or_create_collection(
            name="wiki_titles",
            embedding_function=openai_ef,
        )
        
        # Index missing pages
        self._init_index()
        
        # Load questions
        print("Loading trivia questions...")
        questions = load_dataset(questions_dataset, split="train")
        self.questions = [{"question": q["question"], "answer": q["answer"]} 
                          for q in questions]
        print(f"  Loaded {len(self.questions)} questions")
    
    def _init_index(self):
        """Index any missing pages into ChromaDB."""
        all_ids = list(self.page_id_to_title.keys())
        existing = set()
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i:i+500]
            got = self.collection.get(ids=batch)
            existing.update(got.get("ids", []))
        
        missing = [pid for pid in all_ids if pid not in existing]
        if missing:
            print(f"  Indexing {len(missing)} new pages...")
            documents = [str(self.page_id_to_title[pid]).strip() for pid in missing]
            metadatas = [{"title": self.page_id_to_title[pid]} for pid in missing]
            
            for i in range(0, len(missing), 100):
                self.collection.upsert(
                    ids=missing[i:i+100],
                    documents=documents[i:i+100],
                    metadatas=metadatas[i:i+100],
                )
    
    def search_pages(self, query: str) -> list[dict]:
        """Search for relevant pages by title."""
        results = self.collection.query(query_texts=[query], n_results=10)
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "page_id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
            })
        return output
    
    def view_sections(self, page_id: str) -> list[dict]:
        """Get sections of a page."""
        content = self.page_id_to_content.get(page_id, "")
        sections = []
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                section_id = f"{page_id}:{section_name.lower().replace(' ', '_')}"
                sections.append({
                    "section_id": section_id,
                    "section_name": section_name,
                })
        
        if not sections:
            sections.append({
                "section_id": f"{page_id}:full",
                "section_name": "Full Page",
            })
        
        return sections
    
    def read_section(self, section_id: str) -> str:
        """Read a section's content."""
        if ":" not in section_id:
            return "Invalid section_id format"
        
        page_id, section_name_id = section_id.split(":", 1)
        content = self.page_id_to_content.get(page_id, "")
        lines = content.split("\n")
        
        if section_name_id == "full":
            return content[:2000]  # Limit length
        
        # Find section
        section_start = None
        section_end = None
        
        for i, line in enumerate(lines):
            if line.startswith("#"):
                current = line.lstrip("#").strip().lower().replace(" ", "_")
                if current == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break
        
        if section_start is not None:
            section_end = section_end or len(lines)
            return "\n".join(lines[section_start:section_end])[:2000]
        
        return "Section not found"
    
    def get_train_questions(self, n=None):
        """Get training questions (first 80%)."""
        split = int(len(self.questions) * 0.8)
        qs = self.questions[:split]
        return qs[:n] if n else qs
    
    def get_eval_questions(self, n=None):
        """Get eval questions (last 20%)."""
        split = int(len(self.questions) * 0.8)
        qs = self.questions[split:]
        return qs[:n] if n else qs


# =============================================================================
# MODEL + CARTRIDGE
# =============================================================================

def load_model(model_name):
    """Load model and tokenizer."""
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
    
    config = model.config
    model_info = {
        "num_layers": config.num_hidden_layers,
        "num_kv_heads": config.num_key_value_heads,
        "head_dim": config.hidden_size // config.num_attention_heads,
    }
    
    print(f"  Layers: {model_info['num_layers']}, KV Heads: {model_info['num_kv_heads']}")
    
    return model, tokenizer, model_info


def create_empty_cartridge():
    """Create empty cartridge."""
    return {"keys": [], "values": [], "num_tokens": 0}


def get_cartridge_tensors(cartridge):
    """Get concatenated cartridge tensors."""
    if cartridge["num_tokens"] == 0:
        return None, None
    keys = torch.cat(cartridge["keys"], dim=2)
    values = torch.cat(cartridge["values"], dim=2)
    return keys, values


def freeze_cartridge(cartridge):
    """Freeze cartridge parameters."""
    for i in range(len(cartridge["keys"])):
        cartridge["keys"][i] = cartridge["keys"][i].detach().clone()
        cartridge["values"][i] = cartridge["values"][i].detach().clone()
    return cartridge


def get_kv_cache_from_text(model, tokenizer, text, num_tokens):
    """Get KV cache from processing text."""
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


# =============================================================================
# TOOL CALLING
# =============================================================================

def parse_tool_calls(text):
    """Parse tool calls from model output. Expects format: TOOL: name(args)"""
    calls = []
    
    # Pattern: TOOL: function_name(arg1, arg2, ...)
    pattern = r'TOOL:\s*(\w+)\(([^)]*)\)'
    matches = re.findall(pattern, text)
    
    for name, args_str in matches:
        # Parse arguments (simple string args)
        args = [a.strip().strip('"\'') for a in args_str.split(',') if a.strip()]
        calls.append({"name": name, "args": args})
    
    return calls


def execute_tool(corpus, name, args):
    """Execute a tool and return result."""
    try:
        if name == "search_pages" and len(args) >= 1:
            return json.dumps(corpus.search_pages(args[0]), indent=2)
        elif name == "view_sections" and len(args) >= 1:
            return json.dumps(corpus.view_sections(args[0]), indent=2)
        elif name == "read_section" and len(args) >= 1:
            return corpus.read_section(args[0])
        else:
            return f"Unknown tool or missing args: {name}({args})"
    except Exception as e:
        return f"Tool error: {e}"


# =============================================================================
# GENERATION WITH CARTRIDGE + TOOLS
# =============================================================================

def generate_with_cartridge(model, tokenizer, model_info, messages, cartridge, 
                            max_tokens=200, temperature=0.7):
    """Generate one response with cartridge prefix using manual token-by-token loop."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)
    
    cart_keys, cart_values = get_cartridge_tensors(cartridge)
    
    generated_ids = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Build KV cache with cartridge prefix (rebuilt each step for simplicity)
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


def run_rollout(model, tokenizer, model_info, corpus, question, skill_text, cartridge, max_turns=5):
    """Run a multi-turn rollout with tools. Returns full trace for logging."""
    
    system = f"""{skill_text}

When you need to use a tool, write: TOOL: function_name("argument")
After getting results, provide your final answer.

Example:
TOOL: search_pages("ancient rome")
[results appear]
TOOL: read_section("rome_123:history")
[content appears]
The answer is: Julius Caesar"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\n\nUse the tools to find the answer."}
    ]
    
    # Trace: log each turn in detail
    trace = {
        "question": question,
        "system_prompt": system,
        "turns": [],
    }
    
    for turn in range(max_turns):
        turn_log = {"turn": turn + 1}
        
        # Log what the model sees
        turn_log["input_messages"] = [{"role": m["role"], "content": m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"]} for m in messages]
        
        response = generate_with_cartridge(
            model, tokenizer, model_info, messages, cartridge, max_tokens=300
        )
        messages.append({"role": "assistant", "content": response})
        
        # Log model output
        turn_log["model_output"] = response
        
        # Check for tool calls
        tool_calls = parse_tool_calls(response)
        turn_log["tool_calls_parsed"] = tool_calls
        
        if not tool_calls:
            turn_log["status"] = "no_tools_done"
            trace["turns"].append(turn_log)
            break
        
        # Execute tools
        tool_results = []
        tool_results_log = []
        for call in tool_calls:
            result = execute_tool(corpus, call["name"], call["args"])
            tool_results.append(f"[{call['name']}]: {result}")
            tool_results_log.append({
                "tool": call["name"],
                "args": call["args"],
                "result": result[:500] + "..." if len(result) > 500 else result,
            })
        
        turn_log["tool_results"] = tool_results_log
        turn_log["status"] = "tools_executed"
        trace["turns"].append(turn_log)
        
        messages.append({"role": "user", "content": "\n\n".join(tool_results)})
    
    # Extract final answer (last assistant message)
    final_response = messages[-1]["content"] if messages[-1]["role"] == "assistant" else ""
    
    trace["final_response"] = final_response
    trace["num_turns"] = len([m for m in messages if m["role"] == "assistant"])
    
    return {
        "messages": messages,
        "final_response": final_response,
        "num_turns": trace["num_turns"],
        "trace": trace,
    }


# =============================================================================
# VERIFIERS JUDGE RUBRIC
# =============================================================================

def create_judge_rubric(judge_model="gpt-4.1"):
    """Create a verifiers JudgeRubric for scoring answers."""
    judge_client = AsyncOpenAI()
    
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=vf.Parser(),
        judge_prompt=JUDGE_PROMPT,
    )
    return rubric


async def judge_answer_async(rubric, question, answer, response):
    """Judge if response is correct using verifiers JudgeRubric."""
    # Build prompt and completion in verifiers format
    prompt = [{"role": "user", "content": question}]
    completion = [{"role": "assistant", "content": response}]
    
    # Create minimal state dict for judge
    state = {}
    
    # Call the judge
    judge_response = await rubric.judge(prompt, completion, answer, state)
    
    return "yes" in judge_response.lower()


def judge_answer(rubric, question, answer, response):
    """Sync wrapper for judge_answer_async."""
    return asyncio.run(judge_answer_async(rubric, question, answer, response))


# =============================================================================
# OPENAI HELPER (for skill updates)
# =============================================================================

def call_openai(messages, model="gpt-4.1", temperature=0.3, max_tokens=500):
    """Call OpenAI API."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def update_skill(current_skill, examples, max_tokens=512, tokenizer=None):
    """Update skill based on examples."""
    
    examples_text = ""
    for ex in examples:
        examples_text += f"\nQuestion: {ex['question'][:100]}\n"
        examples_text += f"Correct: {'YES' if ex['correct'] else 'NO'}\n"
        examples_text += f"Turns used: {ex['num_turns']}\n"
        if not ex['correct']:
            examples_text += f"Response snippet: {ex['response'][:200]}...\n"
    
    prompt = f"""You are improving a skill file that guides an LLM to answer trivia questions using Wikipedia search tools.

Current skill:
---
{current_skill}
---

Recent performance:
{examples_text}

Improve the skill to help the model:
1. Find answers more reliably
2. Use fewer tool calls when possible
3. Avoid common mistakes seen above

Keep it concise (~{max_tokens} tokens max). Focus on actionable strategies.

Output ONLY the new skill text:"""

    new_skill = call_openai([{"role": "user", "content": prompt}], model="gpt-4.1", temperature=0.7)
    
    # Truncate if needed
    if tokenizer:
        tokens = tokenizer.encode(new_skill, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            new_skill = tokenizer.decode(tokens)
    
    return new_skill.strip()


# =============================================================================
# CARTRIDGE TRAINING
# =============================================================================

def train_cartridge_step(model, tokenizer, model_info, trainable_keys, trainable_values,
                         frozen_cartridge, skill_text, question, optimizer):
    """One step of cartridge distillation."""
    
    # Teacher: model + full skill
    teacher_messages = [
        {"role": "system", "content": skill_text},
        {"role": "user", "content": f"How would you search for: {question}"}
    ]
    teacher_text = tokenizer.apply_chat_template(teacher_messages, tokenize=False, add_generation_prompt=True)
    teacher_ids = tokenizer(teacher_text, return_tensors="pt").input_ids.to(DEVICE)
    
    # Generate teacher answer
    with torch.no_grad():
        teacher_output = model.generate(
            teacher_ids,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer_ids = teacher_output[0, teacher_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    # Get teacher logits
    with torch.no_grad():
        teacher_out = model(input_ids=teacher_output, use_cache=False)
        teacher_logits = teacher_out.logits[0, teacher_ids.shape[1]-1:-1, :]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        topk_probs, topk_ids = torch.topk(teacher_probs, k=20, dim=-1)
    
    # Student: model + cartridge (no skill in text)
    student_messages = [{"role": "user", "content": f"How would you search for: {question}"}]
    student_text = tokenizer.apply_chat_template(student_messages, tokenize=False, add_generation_prompt=True)
    student_full = student_text + answer_text
    student_ids = tokenizer(student_full, return_tensors="pt").input_ids.to(DEVICE)
    student_prompt_len = len(tokenizer(student_text).input_ids)
    
    # Build student cache with cartridge
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
    
    position_ids = torch.arange(student_ids.shape[1], device=DEVICE).unsqueeze(0) + cart_len
    
    # Student forward
    student_out = model(
        input_ids=student_ids,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )
    
    student_logits = student_out.logits[0, student_prompt_len-1:-1, :]
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    
    # Align lengths
    num_tokens = min(student_logits.shape[0], topk_probs.shape[0])
    
    # Sparse cross-entropy
    student_topk_logprobs = student_log_probs[:num_tokens].gather(
        dim=-1, index=topk_ids[:num_tokens]
    )
    
    loss = -(topk_probs[:num_tokens] * student_topk_logprobs).sum(dim=-1).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="wiki_engram_run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--skill-rounds", type=int, default=5, help="Skill refinement rounds per iteration")
    parser.add_argument("--cartridge-steps", type=int, default=30, help="Cartridge training steps")
    parser.add_argument("--tokens-per-iter", type=int, default=32, help="New cartridge tokens per iteration")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--num-eval", type=int, default=5, help="Number of eval questions per eval")
    parser.add_argument("--initial-eval", action="store_true", help="Run eval before any training")
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    for d in [output_dir, output_dir/"skills", output_dir/"logs", output_dir/"cartridges"]:
        d.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir/"config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 60)
    print("WIKI ENGRAM: Skill + Cartridge Learning for Wiki Search")
    print("Using verifiers.JudgeRubric for answer scoring")
    print("=" * 60)
    
    # Load components
    corpus = WikiCorpus()
    model, tokenizer, model_info = load_model(args.model)
    
    # Create verifiers judge rubric
    print("Setting up verifiers.JudgeRubric...")
    judge_rubric = create_judge_rubric(judge_model="gpt-4.1")
    
    # Freeze model
    for param in model.parameters():
        param.requires_grad = False
    
    # Initialize
    current_skill = INITIAL_SKILL
    cartridge = create_empty_cartridge()
    metrics_history = []
    
    # Save initial
    with open(output_dir/"skills"/"skill_iter_0.md", "w") as f:
        f.write(current_skill)
    
    print(f"\nStarting {args.iterations} iterations")
    print(f"  Skill rounds: {args.skill_rounds}")
    print(f"  Cartridge steps: {args.cartridge_steps}")
    print(f"  Tokens per iter: {args.tokens_per_iter}")
    print(f"  Eval questions: {args.num_eval}")
    
    # =================================================================
    # INITIAL EVAL (before any training)
    # =================================================================
    if args.initial_eval:
        print(f"\n{'='*60}")
        print(f"INITIAL EVAL (before training)")
        print(f"{'='*60}")
        print(f"Testing on {args.num_eval} questions with base skill (no cartridge)...")
        
        eval_qs = corpus.get_eval_questions(args.num_eval)
        initial_traces = []
        correct = 0
        
        for idx, q in enumerate(eval_qs):
            print(f"  [{idx+1}/{len(eval_qs)}] {q['question'][:50]}...")
            
            rollout = run_rollout(
                model, tokenizer, model_info, corpus,
                q["question"], current_skill, cartridge
            )
            is_correct = judge_answer(judge_rubric, q["question"], q["answer"], rollout["final_response"])
            if is_correct:
                correct += 1
            
            initial_traces.append({
                "eval_idx": idx,
                "phase": "initial_eval",
                "ground_truth": q["answer"],
                "correct": is_correct,
                "trace": rollout["trace"],
            })
            
            print(f"      {'✓' if is_correct else '✗'} ({rollout['num_turns']} turns)")
        
        initial_acc = correct / len(eval_qs)
        print(f"\n  Initial accuracy: {initial_acc:.0%} ({correct}/{len(eval_qs)})")
        
        # Save initial eval
        with open(output_dir/"logs"/"initial_eval.json", "w") as f:
            json.dump({
                "accuracy": initial_acc,
                "correct": correct,
                "total": len(eval_qs),
                "traces": initial_traces,
            }, f, indent=2)
        
        # Add to metrics
        metrics_history.append({
            "iteration": 0,
            "phase_a_accuracy": None,
            "eval_accuracy": initial_acc,
            "cartridge_tokens": 0,
            "final_loss": None,
            "time_seconds": 0,
        })
        
        with open(output_dir/"metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
    
    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{args.iterations}")
        print(f"{'='*60}")
        
        # Reset skill each iteration (cartridge accumulates)
        current_skill = INITIAL_SKILL
        
        # =================================================================
        # PHASE A: Skill Refinement
        # =================================================================
        print(f"\n[PHASE A] Skill Refinement ({args.skill_rounds} rounds)")
        
        examples = []
        traces = []  # Full traces for logging
        train_qs = corpus.get_train_questions()
        
        for round_idx in range(args.skill_rounds):
            q = random.choice(train_qs)
            print(f"  Round {round_idx+1}: {q['question'][:50]}...")
            
            rollout = run_rollout(
                model, tokenizer, model_info, corpus,
                q["question"], current_skill, cartridge
            )
            
            correct = judge_answer(judge_rubric, q["question"], q["answer"], rollout["final_response"])
            
            examples.append({
                "question": q["question"],
                "answer": q["answer"],
                "response": rollout["final_response"],
                "correct": correct,
                "num_turns": rollout["num_turns"],
            })
            
            # Save full trace
            traces.append({
                "round": round_idx + 1,
                "phase": "A",
                "ground_truth": q["answer"],
                "correct": correct,
                "trace": rollout["trace"],
            })
            
            print(f"    {'✓' if correct else '✗'} ({rollout['num_turns']} turns)")
        
        # Update skill
        current_skill = update_skill(current_skill, examples, max_tokens=512, tokenizer=tokenizer)
        
        with open(output_dir/"skills"/f"skill_iter_{iteration}_after_a.md", "w") as f:
            f.write(current_skill)
        
        phase_a_acc = sum(1 for e in examples if e["correct"]) / len(examples)
        print(f"  Phase A accuracy: {phase_a_acc:.0%}")
        
        # =================================================================
        # PHASE B: Cartridge Training
        # =================================================================
        print(f"\n[PHASE B] Cartridge Training ({args.cartridge_steps} steps)")
        
        cartridge = freeze_cartridge(cartridge)
        
        # Init new trainable tokens
        new_keys, new_values = get_kv_cache_from_text(
            model, tokenizer, current_skill, args.tokens_per_iter
        )
        trainable_keys = torch.nn.Parameter(new_keys.clone())
        trainable_values = torch.nn.Parameter(new_values.clone())
        
        optimizer = torch.optim.Adam([trainable_keys, trainable_values], lr=2e-2)
        
        losses = []
        for step in range(args.cartridge_steps):
            q = random.choice(train_qs)
            loss = train_cartridge_step(
                model, tokenizer, model_info,
                trainable_keys, trainable_values,
                cartridge, current_skill, q["question"], optimizer
            )
            losses.append(loss)
            
            if (step + 1) % 10 == 0:
                print(f"  Step {step+1}: loss = {sum(losses[-10:])/10:.4f}")
        
        # Add to cartridge
        cartridge["keys"].append(trainable_keys.detach())
        cartridge["values"].append(trainable_values.detach())
        cartridge["num_tokens"] += args.tokens_per_iter
        
        print(f"  Cartridge: {cartridge['num_tokens']} tokens")
        
        # =================================================================
        # EVAL
        # =================================================================
        eval_acc = None
        eval_traces = []
        if iteration % args.eval_every == 0:
            print(f"\n[EVAL] Testing on {args.num_eval} questions...")
            eval_qs = corpus.get_eval_questions(args.num_eval)
            
            correct = 0
            for idx, q in enumerate(eval_qs):
                rollout = run_rollout(
                    model, tokenizer, model_info, corpus,
                    q["question"], current_skill, cartridge
                )
                is_correct = judge_answer(judge_rubric, q["question"], q["answer"], rollout["final_response"])
                if is_correct:
                    correct += 1
                
                # Save eval trace
                eval_traces.append({
                    "eval_idx": idx,
                    "phase": "eval",
                    "ground_truth": q["answer"],
                    "correct": is_correct,
                    "trace": rollout["trace"],
                })
            
            eval_acc = correct / len(eval_qs)
            print(f"  Eval accuracy: {eval_acc:.0%} ({correct}/{len(eval_qs)})")
        
        # =================================================================
        # SAVE EVERYTHING
        # =================================================================
        
        # Save cartridge
        torch.save({
            "keys": cartridge["keys"],
            "values": cartridge["values"],
            "num_tokens": cartridge["num_tokens"],
        }, output_dir/"cartridges"/f"cartridge_{iteration:04d}.pt")
        
        # Save skill
        with open(output_dir/"skills"/f"skill_iter_{iteration}.md", "w") as f:
            f.write(current_skill)
        
        # Save full traces for this iteration
        iter_traces = {
            "iteration": iteration,
            "phase_a_traces": traces,
            "eval_traces": eval_traces,
        }
        with open(output_dir/"logs"/f"traces_iter_{iteration:04d}.json", "w") as f:
            json.dump(iter_traces, f, indent=2)
        
        # Save metrics
        metrics = {
            "iteration": iteration,
            "phase_a_accuracy": phase_a_acc,
            "eval_accuracy": eval_acc,
            "cartridge_tokens": cartridge["num_tokens"],
            "final_loss": losses[-1] if losses else None,
            "time_seconds": time.time() - iter_start,
        }
        metrics_history.append(metrics)
        
        with open(output_dir/"metrics_history.json", "w") as f:
            json.dump(metrics_history, f, indent=2)
        
        print(f"\n  Iteration complete in {metrics['time_seconds']:.1f}s")
        print(f"  Full traces saved to: logs/traces_iter_{iteration:04d}.json")
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final cartridge: {cartridge['num_tokens']} tokens")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

