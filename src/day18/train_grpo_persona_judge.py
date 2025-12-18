"""
Day 19: GRPO with persona judges (via vLLM)
------------------------------------------
Train a local policy model to write a short tweet about a fixed subject.
Reward comes from a round-robin tournament: a demographic-sliced set of personas
votes pairwise on which tweet they'd be more likely to like + retweet.

Design goals:
- single-file, readable, minimal dependencies
- policy model runs locally (HF transformers)
- judge model is served behind an OpenAI-compatible vLLM endpoint
- full per-step JSONL logs (all candidates, matchups, vote counts, Elo ratings)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
import numpy as np
import openai
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


###############################################################################
# Repro / utilities
###############################################################################


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


###############################################################################
# Token logprobs (copied from TRL-style GRPO utility)
###############################################################################


def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if logits.dtype in (torch.float32, torch.float64):
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(row, dim=-1) for row in logits])
        return selected_logits - logsumexp_values

    per_token_logps = []
    for row_logits, row_labels in zip(logits, index):
        row_logps = F.log_softmax(row_logits, dim=-1)
        per_token_logps.append(row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(per_token_logps)


def get_per_token_logps(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    # +1 because we drop last logit (next-token pred)
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    return selective_log_softmax(logits, input_ids)


###############################################################################
# GRPO loss (DR-GRPO variant used in Day 7)
###############################################################################


def compute_grpo_loss(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    max_completion_length: int,
) -> torch.Tensor:
    tokens_to_keep = completion_ids.size(1)
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    logps = get_per_token_logps(model, input_ids, attention_mask, tokens_to_keep)

    # -exp(logp - stopgrad(logp)) * advantage
    if advantages.dim() == 1:
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages.unsqueeze(1)
    else:
        per_token_loss = -torch.exp(logps - logps.detach()) * advantages

    completion_only_mask = completion_mask[:, -tokens_to_keep:]
    return (per_token_loss * completion_only_mask).sum() / (per_token_loss.size(0) * max_completion_length)


###############################################################################
# Policy model: local generation (K candidates per step)
###############################################################################


def load_policy(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def build_tweet_prompt(subject: str) -> tuple[str, str]:
    system = "You are a world-class social media writer. You always write in English."
    user = (
        f"Write a 2-3 sentence tweet about this subject:\n\n{subject}\n\n"
        "Constraints:\n"
        "- Write in English only\n"
        "- Be clear and punchy\n"
        "- No hashtags\n"
        "- Output ONLY the tweet text (no quotes, no labels, no explanations)\n"
    )
    return system, user


def format_prompt(tokenizer: AutoTokenizer, system: str, user: str) -> tuple[str, torch.Tensor, torch.Tensor]:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=True)
    return prompt_text, inputs["input_ids"], inputs["attention_mask"]


@dataclass
class GenerationConfig:
    candidates_per_step: int
    temperature: float
    top_p: float
    max_new_tokens: int


def generate_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    gen_cfg: GenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    k = gen_cfg.candidates_per_step
    prompt_ids = prompt_ids.repeat(k, 1).to(model.device)
    prompt_mask = prompt_mask.repeat(k, 1).to(model.device)

    generation_kwargs = {
        "max_new_tokens": gen_cfg.max_new_tokens,
        "do_sample": True,
        "temperature": gen_cfg.temperature,
        "top_p": gen_cfg.top_p,
        "repetition_penalty": 1.0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    with torch.inference_mode():
        prompt_completion_ids = model.generate(prompt_ids, attention_mask=prompt_mask, **generation_kwargs)

    prompt_len = prompt_ids.size(1)
    prompt_ids_out = prompt_completion_ids[:, :prompt_len]
    completion_ids = prompt_completion_ids[:, prompt_len:]

    # mask completion up to EOS
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=model.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
    seq_idx = torch.arange(is_eos.size(1), device=model.device).expand_as(is_eos)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()

    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    texts = [t.strip() for t in texts]

    return prompt_completion_ids, prompt_ids_out, completion_ids, attention_mask, completion_mask, texts


###############################################################################
# Personas: filtering + sampling
###############################################################################


def build_persona_prompt(persona: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append(
        f"You ARE {persona.get('persona', 'a real person')}. "
        "This is not roleplay - you literally ARE this person with these exact life experiences, beliefs, biases, and worldview."
    )

    for k, label in [
        ("cultural_background", "Your background"),
        ("professional_persona", "Your work life"),
        ("hobbies_and_interests", "What you do for fun"),
        ("arts_persona", "Your taste in arts & culture"),
        ("sports_persona", "Your sports & fitness"),
        ("culinary_persona", "Your food preferences"),
        ("travel_persona", "How you travel"),
        ("skills_and_expertise", "Your skills"),
        ("career_goals_and_ambitions", "Your ambitions"),
    ]:
        v = persona.get(k)
        if v:
            parts.append(f"\n{label}: {v}")

    demo = []
    if persona.get("age"):
        demo.append(f"{persona['age']} years old")
    if persona.get("sex"):
        demo.append(str(persona["sex"]).lower())
    if persona.get("marital_status"):
        demo.append(str(persona["marital_status"]).replace("_", " "))
    if persona.get("education_level"):
        demo.append(f"education: {str(persona['education_level']).replace('_', ' ')}")
    if persona.get("occupation") and persona["occupation"] not in ("no_occupation", "not_in_workforce"):
        demo.append(f"works as: {str(persona['occupation']).replace('_', ' ')}")
    if persona.get("city") and persona.get("state"):
        demo.append(f"lives in {persona['city']}, {persona['state']}")
    if demo:
        parts.append(f"\nYou are: {', '.join(demo)}")

    return "\n".join(parts)


def _passes_filters(persona: dict[str, Any], filters: dict[str, Any]) -> bool:
    def norm(field: str, x: Any) -> Any:
        if not isinstance(x, str):
            return x
        s = x.strip()
        s_low = s.lower()

        # Common aliases to keep configs ergonomic
        if field == "education_level":
            aliases = {
                "bachelors_degree": "bachelors",
                "bachelor_degree": "bachelors",
                "bachelor's": "bachelors",
                "masters_degree": "masters",
                "master_degree": "masters",
                "master's": "masters",
                "phd": "doctorate",
                "doctorate_degree": "doctorate",
            }
            s_low = aliases.get(s_low, s_low)

        return s_low

    for field, rule in (filters or {}).items():
        val = persona.get(field)
        if rule is None:
            continue

        if isinstance(rule, dict) and ("min" in rule or "max" in rule):
            if val is None:
                return False
            if "min" in rule and val < rule["min"]:
                return False
            if "max" in rule and val > rule["max"]:
                return False
            continue

        if isinstance(rule, list):
            v = norm(field, val)
            allowed = [norm(field, r) for r in rule]
            if v not in allowed:
                return False
            continue

        # exact match fallback
        if norm(field, val) != norm(field, rule):
            return False

    return True


def _persona_meta(persona: dict[str, Any], idx: int) -> dict[str, Any]:
    # Keep this small-ish; expand later as needed for frontend drilldown.
    return {
        "idx": idx,
        "uuid": persona.get("uuid", ""),
        "age": persona.get("age"),
        "sex": persona.get("sex"),
        "education_level": persona.get("education_level"),
        "occupation": persona.get("occupation"),
        "city": persona.get("city"),
        "state": persona.get("state"),
        "zipcode": persona.get("zipcode"),
    }


def find_eligible_indices(
    dataset: Any,
    filters: Optional[dict[str, Any]],
    *,
    desc: str,
) -> list[int]:
    # This is the slow part (1M scan). Show a progress bar and how many matches we’ve found.
    eligible: list[int] = []
    filters = filters or {}

    debug_counts: dict[str, dict[str, int]] = {k: {} for k in filters.keys()}
    bar = tqdm(range(len(dataset)), desc=desc, mininterval=1.0)
    for i in bar:
        p = dataset[i]

        for field in debug_counts.keys():
            v = p.get(field)
            if isinstance(v, str):
                v = v.strip()
            debug_counts[field][str(v)] = debug_counts[field].get(str(v), 0) + 1

        if _passes_filters(p, filters):
            eligible.append(i)
            # Update progress bar occasionally (cheap).
            if len(eligible) % 5000 == 0:
                bar.set_postfix({"eligible": len(eligible)})

    bar.set_postfix({"eligible": len(eligible)})
    bar.close()

    if len(eligible) == 0 and filters:
        hint_parts = []
        for field, counts in debug_counts.items():
            top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
            hint_parts.append({field: [k for k, _ in top]})
        raise ValueError(
            "0 eligible personas for the requested filters. "
            f"filters={filters}. "
            f"Top observed values per filtered field (debug)={hint_parts}"
        )

    return eligible


def sample_from_eligible(eligible: list[int], num_personas: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if len(eligible) < num_personas:
        raise ValueError(f"Not enough eligible personas: need {num_personas}, found {len(eligible)}")
    eligible = list(eligible)
    rng.shuffle(eligible)
    return eligible[:num_personas]


###############################################################################
# vLLM judge client (pairwise A/B)
###############################################################################


def parse_choice(text: str) -> Optional[str]:
    """Parse A or B from model output. Returns None if can't parse."""
    if not text:
        return None
    text = text.strip()
    
    # Try \boxed{A} or \boxed{B} first
    m = re.search(r"\\boxed\{([AB])\}", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Try just A or B at the start of the response
    m = re.match(r"^([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Try "Tweet A" or "Tweet B" anywhere
    m = re.search(r"\btweet\s*([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Try just A or B anywhere as a standalone word
    m = re.search(r"\b([AB])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    return None


def parse_yesno(text: str) -> Optional[str]:
    """Parse Y or N from model output. Returns None if can't parse."""
    if not text:
        return None
    text = text.strip()
    
    # Try \boxed{Y} or \boxed{N}
    m = re.search(r"\\boxed\{([YN])\}", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    # Try Yes/No at start
    if text.lower().startswith("yes"):
        return "Y"
    if text.lower().startswith("no"):
        return "N"
    
    # Try just Y or N at start
    m = re.match(r"^([YN])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    
    return None


@dataclass
class JudgeConfig:
    base_url: str
    model: str
    max_concurrent: int
    timeout_s: int


class VLLMJudge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg
        self._sema = asyncio.Semaphore(cfg.max_concurrent)

    async def _chat(self, session: aiohttp.ClientSession, messages: list[dict[str, str]], max_tokens: int = 64) -> Optional[str]:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        try:
            async with self._sema:
                # Timeout applies AFTER acquiring semaphore (not while waiting in queue)
                req_timeout = aiohttp.ClientTimeout(total=self.cfg.timeout_s)
                async with session.post(f"{self.cfg.base_url}/chat/completions", json=payload, timeout=req_timeout) as resp:
                    data = await resp.json()
            return data["choices"][0]["message"]["content"]
        except asyncio.TimeoutError:
            if not hasattr(self, "_timeout_count"):
                self._timeout_count = 0
            self._timeout_count += 1
            if self._timeout_count <= 3:
                print(f"[WARN] Request timed out ({self._timeout_count} so far)")
            return None
        except Exception as e:
            if not hasattr(self, "_error_count"):
                self._error_count = 0
            self._error_count += 1
            if self._error_count <= 3:
                print(f"[WARN] Request failed: {type(e).__name__}: {e}")
            return None

    async def vote_pairwise(
        self,
        session: aiohttp.ClientSession,
        persona: dict[str, Any],
        subject: str,
        tweet_a: str,
        tweet_b: str,
        debug: bool = False,
    ) -> Optional[str]:
        persona_desc = build_persona_prompt(persona)
        prompt = (
            f"{persona_desc}\n\n---\n\n"
            f"You're scrolling social media. You see two tweets about:\n{subject}\n\n"
            "Which tweet would you be more likely to LIKE and RETWEET?\n\n"
            f"TWEET A:\n{tweet_a}\n\n"
            f"TWEET B:\n{tweet_b}\n\n"
            "Answer with a single letter: A or B"
        )
        out = await self._chat(session, [{"role": "user", "content": prompt}], max_tokens=8)
        if out is None:
            return None
        choice = parse_choice(out)
        if debug and choice is None:
            print(f"[DEBUG] Could not parse choice from: {repr(out)}")
        return choice

    async def yesno_like_retweet(
        self,
        session: aiohttp.ClientSession,
        persona: dict[str, Any],
        subject: str,
        tweet: str,
    ) -> Optional[str]:
        persona_desc = build_persona_prompt(persona)
        prompt = (
            f"{persona_desc}\n\n---\n\n"
            f"You're scrolling social media. You see this tweet about:\n{subject}\n\n"
            f"TWEET:\n{tweet}\n\n"
            "Would you be likely to LIKE and RETWEET this?\n"
            "Reply with ONLY one character: \\boxed{Y} or \\boxed{N}."
        )
        out = await self._chat(session, [{"role": "user", "content": prompt}], max_tokens=16)
        if out is None:
            return None
        return parse_yesno(out)


def generate_gpt41_tweet(subject: str) -> str:
    """Generate a single tweet from GPT-4.1 via OpenAI API."""
    client = openai.OpenAI()  # Uses OPENAI_API_KEY from env
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a world-class social media writer."},
            {"role": "user", "content": (
                f"Write a 2-3 sentence tweet about: {subject}\n\n"
                "Be clear and punchy. No hashtags. Output ONLY the tweet text."
            )},
        ],
        max_tokens=128,
        temperature=0.9,
    )
    return response.choices[0].message.content.strip()


###############################################################################
# Tournament + Elo
###############################################################################


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def run_elo_tournament(
    k_candidates: int,
    matchups: list[tuple[int, int]],
    matchup_winners: list[int],
    k_factor: float,
    init_rating: float = 1000.0,
) -> list[float]:
    ratings = [init_rating for _ in range(k_candidates)]
    for (i, j), winner in zip(matchups, matchup_winners):
        ra, rb = ratings[i], ratings[j]
        ea, eb = _elo_expected(ra, rb), _elo_expected(rb, ra)
        sa = 1.0 if winner == i else 0.0
        sb = 1.0 - sa
        ratings[i] = ra + k_factor * (sa - ea)
        ratings[j] = rb + k_factor * (sb - eb)
    return ratings


def compute_win_rates(
    k_candidates: int,
    matchup_logs: list[dict],
) -> list[float]:
    """Compute win rate per candidate from matchup vote counts.
    
    Win rate = total votes received / total votes cast across all matchups involving this candidate.
    These sum to 1.0 across all candidates (excluding failures).
    """
    total_votes_for = [0] * k_candidates  # votes where this candidate was chosen
    total_votes_in = [0] * k_candidates   # votes in matchups involving this candidate
    
    for m in matchup_logs:
        i, j = m["i"], m["j"]
        vi, vj = m["votes_i"], m["votes_j"]
        total_matchup_votes = vi + vj
        
        total_votes_for[i] += vi
        total_votes_for[j] += vj
        total_votes_in[i] += total_matchup_votes
        total_votes_in[j] += total_matchup_votes
    
    # Win rate = votes for me / votes in my matchups
    win_rates = []
    for i in range(k_candidates):
        if total_votes_in[i] > 0:
            win_rates.append(total_votes_for[i] / total_votes_in[i])
        else:
            win_rates.append(0.0)
    
    return win_rates


async def judge_round_robin(
    judge: VLLMJudge,
    personas: list[dict[str, Any]],
    subject: str,
    candidates: list[str],
    seed: int,
    debug: bool = False,
) -> dict[str, Any]:
    k = len(candidates)
    matchups = [(i, j) for i in range(k) for j in range(i + 1, k)]

    connector = aiohttp.TCPConnector(limit=judge.cfg.max_concurrent * 2, limit_per_host=judge.cfg.max_concurrent * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        matchup_logs = []
        winners = []
        total_failures = 0

        for matchup_idx, (i, j) in enumerate(matchups):
            tweet_i, tweet_j = candidates[i], candidates[j]
            
            # IMPORTANT: Randomize presentation order PER PERSONA to eliminate position bias
            # Each persona sees tweets in random order (A/B swap)
            rng = random.Random(seed + matchup_idx * 12345)
            swap_flags = [rng.random() < 0.5 for _ in personas]  # True = swap order
            
            async def vote_with_swap(persona_idx: int) -> Optional[str]:
                p = personas[persona_idx]
                should_debug = debug and matchup_idx == 0 and persona_idx < 3  # Debug first 3 votes of first matchup
                if swap_flags[persona_idx]:
                    # Swapped: j is shown as A, i is shown as B
                    vote = await judge.vote_pairwise(session, p, subject, tweet_j, tweet_i, debug=should_debug)
                    # Map vote back: if they said A, they meant j; if B, they meant i
                    if vote == "A":
                        return "B"  # They picked j, which is "B" in original order
                    elif vote == "B":
                        return "A"  # They picked i, which is "A" in original order
                    return vote  # None
                else:
                    # Normal order: i is A, j is B
                    return await judge.vote_pairwise(session, p, subject, tweet_i, tweet_j, debug=should_debug)
            
            tasks = [vote_with_swap(idx) for idx in range(len(personas))]
            votes = await asyncio.gather(*tasks)

            # Now votes are normalized: A = tweet i, B = tweet j
            votes_i = sum(1 for v in votes if v == "A")
            votes_j = sum(1 for v in votes if v == "B")
            failures = sum(1 for v in votes if v is None)
            total_failures += failures

            if votes_i == votes_j:
                # deterministic tiebreak by matchup + seed
                tie_rng = random.Random((seed + 1_000_000) ^ (matchup_idx * 9973))
                winner = i if tie_rng.random() < 0.5 else j
            else:
                winner = i if votes_i > votes_j else j

            winners.append(winner)
            vote_details = [
                {"uuid": personas[idx].get("uuid", ""), "choice": v, "was_swapped": swap_flags[idx]} 
                for idx, v in enumerate(votes)
            ]
            matchup_logs.append(
                {
                    "i": i,
                    "j": j,
                    "votes_i": votes_i,
                    "votes_j": votes_j,
                    "failures": failures,
                    "winner": winner,
                    "votes": vote_details,
                }
            )

    return {
        "matchups": matchups,
        "matchup_logs": matchup_logs,
        "matchup_winners": winners,
        "total_failures": total_failures,
    }


async def eval_like_retweet_rate(
    judge: VLLMJudge,
    personas: list[dict[str, Any]],
    subject: str,
    tweet: str,
    *,
    sample_votes_n: int = 0,
    in_target_mask: Optional[list[bool]] = None,
) -> dict[str, Any]:
    connector = aiohttp.TCPConnector(limit=judge.cfg.max_concurrent * 2, limit_per_host=judge.cfg.max_concurrent * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        votes: list[Optional[str]] = []
        # Avoid spawning 10k tasks at once.
        chunk = max(256, judge.cfg.max_concurrent * 4)
        for start in range(0, len(personas), chunk):
            sl = personas[start : start + chunk]
            tasks = [judge.yesno_like_retweet(session, p, subject, tweet) for p in sl]
            votes.extend(await asyncio.gather(*tasks))

    yes = sum(1 for v in votes if v == "Y")
    no = sum(1 for v in votes if v == "N")
    failures = sum(1 for v in votes if v is None)
    total = yes + no
    rate = yes / total if total > 0 else 0.0

    sample_votes: list[dict[str, Any]] = []
    if sample_votes_n > 0:
        for i in range(min(sample_votes_n, len(personas))):
            sample_votes.append(
                {
                    "uuid": personas[i].get("uuid", ""),
                    "vote": votes[i],
                    "in_target_demo": (in_target_mask[i] if in_target_mask is not None else None),
                }
            )

    return {"yes": yes, "no": no, "failures": failures, "yes_rate": rate, "sample_votes": sample_votes}


###############################################################################
# Human-readable summaries
###############################################################################


def format_step_summary(
    step: int,
    candidates: list[str],
    win_rates: list[float],
    matchup_logs: list[dict],
    winner_idx: int,
    loss: float,
    num_judges: int,
) -> str:
    """Format a human-readable summary for a training step."""
    lines = [
        f"\n{'='*70}",
        f"STEP {step}",
        f"{'='*70}",
        "",
        "CANDIDATES (by win rate):",
    ]
    
    # Sort by win rate for display
    sorted_idxs = sorted(range(len(win_rates)), key=lambda i: win_rates[i], reverse=True)
    for rank, i in enumerate(sorted_idxs, 1):
        marker = " ★ WINNER" if i == winner_idx else ""
        lines.append(f"  [{rank}] Win rate: {win_rates[i]*100:.1f}%{marker}")
        tweet = candidates[i]
        lines.append(f"      \"{tweet}\"")
        lines.append("")
    
    lines.append("MATCHUPS:")
    for m in matchup_logs:
        i, j = m["i"], m["j"]
        vi, vj = m["votes_i"], m["votes_j"]
        lines.append(f"  Tweet {i} vs Tweet {j}: {vi}-{vj}")
    
    lines.append("")
    lines.append(f"Judges: {num_judges} | Loss: {loss:.4f}")
    lines.append("")
    
    return "\n".join(lines)


def format_eval_summary(
    step: int,
    tweet: str,
    target_stats: dict,
    general_stats: dict,
) -> str:
    """Format a human-readable summary for an eval step."""
    lines = [
        f"\n{'~'*70}",
        f"EVAL @ STEP {step}",
        f"{'~'*70}",
        "",
        "BEST TWEET:",
        f"  \"{tweet}\"",
        "",
        "RESULTS:",
        f"  Target demographic:  {target_stats['yes']}/{target_stats['yes']+target_stats['no']} would like/retweet ({target_stats['yes_rate']*100:.1f}%)",
        f"  General population:  {general_stats['yes']}/{general_stats['yes']+general_stats['no']} would like/retweet ({general_stats['yes_rate']*100:.1f}%)",
        "",
    ]
    return "\n".join(lines)


###############################################################################
# Main training loop
###############################################################################


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to JSON config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config)
    cfg = _read_json(cfg_path)

    subject: str = cfg["subject"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", cfg)

    seed = int(cfg.get("seed", 0))
    seed_everything(seed)

    # Policy
    policy_cfg = cfg["policy"]
    model, tokenizer = load_policy(policy_cfg["model_name"])
    model.train()

    gen_cfg = GenerationConfig(
        candidates_per_step=int(policy_cfg.get("candidates_per_step", 4)),
        temperature=float(policy_cfg.get("temperature", 0.9)),
        top_p=float(policy_cfg.get("top_p", 1.0)),
        max_new_tokens=int(policy_cfg.get("max_new_tokens", 128)),
    )

    # Optim
    lr = float(policy_cfg.get("learning_rate", 5e-6))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(policy_cfg.get("adam_beta1", 0.9)), float(policy_cfg.get("adam_beta2", 0.99))),
        weight_decay=float(policy_cfg.get("weight_decay", 0.1)),
    )
    warmup_percent = float(policy_cfg.get("warmup_percent", 0.1))
    num_steps = int(policy_cfg.get("num_train_steps", 1000))
    warmup_steps = int(warmup_percent * num_steps)

    def lr_mult(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)
    grad_accum = int(policy_cfg.get("gradient_accumulation_steps", 4))
    max_grad_norm = float(policy_cfg.get("max_grad_norm", 0.1))

    # Judge
    judge_cfg = cfg["judge"]
    judge = VLLMJudge(
        JudgeConfig(
            base_url=str(judge_cfg.get("base_url", "http://localhost:8000/v1")).rstrip("/"),
            model=str(judge_cfg.get("model", "Qwen/Qwen2.5-7B-Instruct")),
            max_concurrent=int(judge_cfg.get("max_concurrent", 128)),
            timeout_s=int(judge_cfg.get("timeout_s", 120)),
        )
    )
    elo_k = float(judge_cfg.get("elo_k", 32.0))

    # Load personas once
    print("Loading nvidia/Nemotron-Personas-USA dataset...")
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
    print(f"Loaded {len(ds):,} personas")

    train_judges_cfg = cfg["train_judges"]
    train_filters = train_judges_cfg.get("filters", {})
    train_num_personas = int(train_judges_cfg.get("num_personas", 50))

    eligible = find_eligible_indices(ds, train_filters, desc="Filtering personas for target demographic")
    # Fixed judge set for the run (as specified)
    train_persona_idxs = sample_from_eligible(eligible, train_num_personas, seed=seed + 101)
    train_personas = [ds[i] for i in train_persona_idxs]

    eval_cfg = cfg.get("eval", {})
    eval_every = int(eval_cfg.get("every_steps", 50))
    eval_general_n = int(eval_cfg.get("general_num_personas", 10_000))
    eval_target_n = int(eval_cfg.get("target_num_personas", 1_000))
    eval_sample_votes_n = int(eval_cfg.get("sample_votes_n", 200))

    # Fixed eval samples (keep stable for a clean curve)
    rng = random.Random(seed + 202)
    general_idxs = rng.sample(range(len(ds)), eval_general_n)
    general_personas = [ds[i] for i in general_idxs]

    target_idxs = sample_from_eligible(eligible, eval_target_n, seed=seed + 303)
    target_personas = [ds[i] for i in target_idxs]

    # Persist persona UUIDs (and membership) so the frontend can drill down later.
    persona_sets = {
        "target_filters": train_filters,
        "train_judges": [_persona_meta(ds[i], i) | {"in_target_demo": True} for i in train_persona_idxs],
        "eval_target": [_persona_meta(ds[i], i) | {"in_target_demo": True} for i in target_idxs],
        "eval_general": [
            _persona_meta(ds[i], i) | {"in_target_demo": _passes_filters(ds[i], train_filters)}
            for i in general_idxs
        ],
    }
    _write_json(output_dir / "persona_sets.json", persona_sets)

    system_prompt, user_prompt = build_tweet_prompt(subject)
    prompt_text, prompt_ids, prompt_mask = format_prompt(tokenizer, system_prompt, user_prompt)

    train_log_path = output_dir / "train_log.jsonl"
    eval_log_path = output_dir / "eval_log.jsonl"
    summary_path = output_dir / "training_summary.txt"
    
    # Write header to summary file
    with summary_path.open("w") as f:
        f.write(f"Day 19 Training Summary\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Target demographic filters: {json.dumps(train_filters)}\n")
        f.write(f"Training judges: {train_num_personas}\n")
        f.write(f"Eval target personas: {eval_target_n}, Eval general personas: {eval_general_n}\n")
        f.write(f"\n")

    optimizer.zero_grad()
    accumulated_loss = 0.0

    for step in tqdm(range(num_steps), desc="Training", disable=False):
        # Generate candidates (no grad)
        with torch.no_grad():
            _, prompt_ids_b, completion_ids, attention_mask, completion_mask, candidates = generate_candidates(
                model, tokenizer, prompt_ids, prompt_mask, gen_cfg
            )

        # Tournament judging (async)
        t0 = time.time()
        rr = asyncio.run(judge_round_robin(judge, train_personas, subject, candidates, seed=seed + step, debug=(step == 0)))
        matchups = rr["matchups"]
        matchup_logs = rr["matchup_logs"]
        matchup_winners = rr["matchup_winners"]
        failures = rr["total_failures"]

        elo = run_elo_tournament(
            k_candidates=len(candidates),
            matchups=matchups,
            matchup_winners=matchup_winners,
            k_factor=elo_k,
        )
        
        # Compute win rates (this is what we use for rewards)
        win_rates = compute_win_rates(len(candidates), matchup_logs)

        winner_idx = int(max(range(len(win_rates)), key=lambda i: win_rates[i]))
        # Use win rates as rewards (they sum to ~1.0, more interpretable than Elo)
        rewards = torch.tensor(win_rates, device=model.device, dtype=torch.float32)

        # Group normalize to advantages (K candidates = 1 group)
        mean = rewards.mean()
        std = rewards.std()
        scalar_adv = (rewards - mean) / (std + 1e-4)
        advantages = scalar_adv.unsqueeze(1)

        # Loss
        loss = compute_grpo_loss(
            model=model,
            prompt_ids=prompt_ids_b,
            completion_ids=completion_ids,
            attention_mask=attention_mask,
            completion_mask=completion_mask,
            advantages=advantages,
            max_completion_length=int(policy_cfg.get("max_new_tokens", 128)),
        )

        (loss / grad_accum).backward()
        accumulated_loss += float(loss.item())

        # Optim step
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
        scheduler.step()

        step_log = {
            "step": step,
            "subject": subject,
            "prompt": prompt_text,
            "candidates": [
                {
                    "idx": i,
                    "text": t,
                    "win_rate": float(win_rates[i]),
                    "elo": float(elo[i]),
                    "advantage": float(scalar_adv[i].item()),
                }
                for i, t in enumerate(candidates)
            ],
            "winner_idx": winner_idx,
            "matchups": matchup_logs,
            "judge_failures": failures,
            "timing": {"judge_s": time.time() - t0},
            "train": {"loss": float(loss.item()), "lr": float(scheduler.get_last_lr()[0])},
        }
        _append_jsonl(train_log_path, step_log)

        # Human-readable summary (print + append to file)
        summary = format_step_summary(
            step=step,
            candidates=candidates,
            win_rates=win_rates,
            matchup_logs=matchup_logs,
            winner_idx=winner_idx,
            loss=float(loss.item()),
            num_judges=len(train_personas),
        )
        tqdm.write(summary)
        with summary_path.open("a") as f:
            f.write(summary + "\n")

        # Periodic eval: Model vs GPT-4.1 round-robin tournament
        if step % eval_every == 0:
            t1 = time.time()
            
            # Generate 8 model tweets
            tqdm.write(f"\n[Eval {step}] Generating 8 model tweets...")
            eval_gen_cfg = GenerationConfig(
                candidates_per_step=8,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                max_new_tokens=gen_cfg.max_new_tokens,
            )
            _, _, _, _, _, model_tweets = generate_candidates(
                model, tokenizer, prompt_ids, prompt_mask, eval_gen_cfg
            )
            
            # Generate 8 GPT-4.1 tweets
            tqdm.write(f"[Eval {step}] Generating 8 GPT-4.1 tweets...")
            gpt_tweets = []
            for _ in range(8):
                try:
                    gt = generate_gpt41_tweet(subject)
                    gpt_tweets.append(gt)
                except Exception as e:
                    gpt_tweets.append(f"[GPT-4.1 failed: {e}]")
            
            # Round-robin: each model tweet vs each GPT tweet = 64 matchups
            # Each persona votes on ONE random matchup (to keep API calls reasonable)
            async def eval_round_robin(personas: list, seed_offset: int, label: str) -> dict:
                """Each persona votes on one random model-vs-gpt matchup."""
                connector = aiohttp.TCPConnector(limit=judge.cfg.max_concurrent)
                
                async with aiohttp.ClientSession(connector=connector) as session:
                    rng = random.Random(seed + step + seed_offset)
                    
                    # Pre-assign each persona to a random matchup
                    assignments = []
                    for _ in personas:
                        mi = rng.randint(0, 7)  # model tweet index
                        gi = rng.randint(0, 7)  # gpt tweet index
                        swap = rng.random() < 0.5  # position swap
                        assignments.append((mi, gi, swap))
                    
                    async def vote_one(idx: int) -> tuple[str, int, int, Optional[str]]:
                        p = personas[idx]
                        uuid = p.get("uuid", "")
                        mi, gi, swap = assignments[idx]
                        mt, gt = model_tweets[mi], gpt_tweets[gi]
                        
                        if swap:
                            raw = await judge.vote_pairwise(session, p, subject, gt, mt)
                            vote = "model" if raw == "B" else ("gpt" if raw == "A" else None)
                        else:
                            raw = await judge.vote_pairwise(session, p, subject, mt, gt)
                            vote = "model" if raw == "A" else ("gpt" if raw == "B" else None)
                        return uuid, mi, gi, vote
                    
                    # Process in batches
                    batch_size = 256
                    all_results = []
                    pbar = tqdm(total=len(personas), desc=f"Eval {label}", leave=False)
                    for batch_start in range(0, len(personas), batch_size):
                        batch_end = min(batch_start + batch_size, len(personas))
                        tasks = [vote_one(i) for i in range(batch_start, batch_end)]
                        batch_results = await asyncio.gather(*tasks)
                        all_results.extend(batch_results)
                        pbar.update(len(batch_results))
                    pbar.close()
                
                model_votes = sum(1 for _, _, _, v in all_results if v == "model")
                gpt_votes = sum(1 for _, _, _, v in all_results if v == "gpt")
                failures = sum(1 for _, _, _, v in all_results if v is None)
                total = model_votes + gpt_votes
                win_rate = model_votes / total if total > 0 else 0.5
                
                return {
                    "model_votes": model_votes,
                    "gpt_votes": gpt_votes,
                    "failures": failures,
                    "win_rate": win_rate,
                    "votes": [{"uuid": uuid, "mi": mi, "gi": gi, "voted_for": v} for uuid, mi, gi, v in all_results],
                }
            
            target_results = asyncio.run(eval_round_robin(target_personas, seed_offset=10000, label="target"))
            general_results = asyncio.run(eval_round_robin(general_personas, seed_offset=20000, label="general"))
            
            # Save eval results
            eval_data = {
                "step": step,
                "model_tweets": model_tweets,
                "gpt_tweets": gpt_tweets,
                "target_demo": {
                    "win_rate": target_results["win_rate"],
                    "model_votes": target_results["model_votes"],
                    "gpt_votes": target_results["gpt_votes"],
                    "failures": target_results["failures"],
                },
                "general_pop": {
                    "win_rate": general_results["win_rate"],
                    "model_votes": general_results["model_votes"],
                    "gpt_votes": general_results["gpt_votes"],
                    "failures": general_results["failures"],
                },
                "timing_s": time.time() - t1,
            }
            
            eval_results_path = output_dir / "eval_results.json"
            all_evals: dict[str, Any] = {}
            if eval_results_path.exists():
                all_evals = _read_json(eval_results_path)
            all_evals[str(step)] = eval_data
            _write_json(eval_results_path, all_evals)
            
            # Save per-vote data for frontend
            eval_votes_path = output_dir / "eval_votes.json"
            all_votes: dict[str, Any] = {}
            if eval_votes_path.exists():
                all_votes = _read_json(eval_votes_path)
            all_votes[str(step)] = {
                "target_votes": target_results["votes"],
                "general_votes": general_results["votes"],
            }
            _write_json(eval_votes_path, all_votes)
            
            # Print summary
            tqdm.write(f"\n{'='*60}")
            tqdm.write(f"EVAL STEP {step}: Model vs GPT-4.1 (8v8 round-robin)")
            tqdm.write(f"{'='*60}")
            tqdm.write(f"TARGET DEMO:  Model {target_results['win_rate']*100:.1f}% win rate")
            tqdm.write(f"              ({target_results['model_votes']} vs {target_results['gpt_votes']}, {target_results['failures']} failures)")
            tqdm.write(f"GENERAL POP:  Model {general_results['win_rate']*100:.1f}% win rate")
            tqdm.write(f"              ({general_results['model_votes']} vs {general_results['gpt_votes']}, {general_results['failures']} failures)")
            tqdm.write(f"{'='*60}\n")

        # Optional checkpoint
        save_every = int(policy_cfg.get("save_every_steps", 50))
        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = output_dir / f"checkpoint_step_{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)


if __name__ == "__main__":
    main()

