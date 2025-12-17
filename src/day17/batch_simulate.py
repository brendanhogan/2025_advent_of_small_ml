"""
Batch Persona Simulation - Process all 1M personas efficiently

Features:
- Async concurrent requests to vLLM
- Periodic checkpointing (saves every N personas)  
- Resume from checkpoint
- Progress tracking

Usage:
    # Start vLLM server (see exps.sh for multi-GPU):
    vllm serve Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 4 --max-num-seqs 512 --port 8000
    
    # Run simulation - crank up --max-concurrent for more throughput:
    uv run python batch_simulate.py --content "Your content" --output run_001 --max-concurrent 256
    
    # Resume from checkpoint:
    uv run python batch_simulate.py --content "Your content" --output run_001 --max-concurrent 256 --resume
"""

import argparse
import asyncio
import json
import time
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import aiohttp
from datasets import load_dataset
from tqdm import tqdm


@dataclass
class PersonaRating:
    """Rating from a single persona."""
    uuid: str
    idx: int  # Index in dataset for resumability
    likeability: int
    emotional_activation: int
    reasoning: str
    sex: str
    age: int
    marital_status: str
    education_level: str
    bachelors_field: str
    occupation: str
    city: str
    state: str
    zipcode: str
    latency_ms: float


def build_persona_prompt(persona: dict) -> str:
    """Build a rich persona description from all fields."""
    parts = []
    
    # Strong identity framing
    parts.append(f"You ARE {persona.get('persona', 'a real person')}. This is not roleplay - you literally ARE this person with these exact life experiences, beliefs, biases, and worldview.")
    
    if persona.get('cultural_background'):
        parts.append(f"\nYour background: {persona['cultural_background']}")
    if persona.get('professional_persona'):
        parts.append(f"\nYour work life: {persona['professional_persona']}")
    if persona.get('hobbies_and_interests'):
        parts.append(f"\nWhat you do for fun: {persona['hobbies_and_interests']}")
    if persona.get('arts_persona'):
        parts.append(f"\nYour taste in arts & culture: {persona['arts_persona']}")
    if persona.get('sports_persona'):
        parts.append(f"\nYour sports & fitness: {persona['sports_persona']}")
    if persona.get('culinary_persona'):
        parts.append(f"\nYour food preferences: {persona['culinary_persona']}")
    if persona.get('travel_persona'):
        parts.append(f"\nHow you travel: {persona['travel_persona']}")
    if persona.get('skills_and_expertise'):
        parts.append(f"\nYour skills: {persona['skills_and_expertise']}")
    if persona.get('career_goals_and_ambitions'):
        parts.append(f"\nYour ambitions: {persona['career_goals_and_ambitions']}")
    
    demo_parts = []
    if persona.get('age'):
        demo_parts.append(f"{persona['age']} years old")
    if persona.get('sex'):
        demo_parts.append(persona['sex'].lower())
    if persona.get('marital_status'):
        demo_parts.append(persona['marital_status'].replace('_', ' '))
    if persona.get('education_level'):
        demo_parts.append(f"education: {persona['education_level'].replace('_', ' ')}")
    if persona.get('occupation') and persona['occupation'] not in ('no_occupation', 'not_in_workforce'):
        demo_parts.append(f"works as: {persona['occupation'].replace('_', ' ')}")
    if persona.get('city') and persona.get('state'):
        demo_parts.append(f"lives in {persona['city']}, {persona['state']}")
    
    if demo_parts:
        parts.append(f"\nYou are: {', '.join(demo_parts)}")
    
    return '\n'.join(parts)


def build_rating_prompt(persona_description: str, content: str) -> str:
    """Build the full prompt for rating content."""
    return f"""{persona_description}

---

You're scrolling through social media and see this post:

\"\"\"{content}\"\"\"

React HONESTLY as yourself - not as a neutral AI. You have opinions, biases, and gut reactions based on your life experiences. Based on your background, values, age, and worldview - do you love this, hate this, or feel something in between?

Think through your reaction, then rate on two scales (1-10):

LIKEABILITY (1-10): How much do you like/agree with this?
  1-2 = Hate it, strongly disagree, offensive
  3-4 = Dislike, disagree  
  5-6 = Neutral, mixed feelings
  7-8 = Like it, agree
  9-10 = Love it, strongly agree, exactly right

EMOTIONAL ACTIVATION (1-10): How emotionally stirred are you?
  1-2 = Don't care at all
  3-4 = Mild interest
  5-6 = Got my attention
  7-8 = Strong feelings, want to engage
  9-10 = Intense reaction, must respond/share

First, briefly explain your gut reaction as this persona (1-2 sentences). Then give your final ratings as \\boxed{{L,E}} where L is likeability and E is emotional activation.

Example format:
As a [your perspective], this [your reaction]. \\boxed{{7,4}}"""


def parse_rating_response(response_text: str) -> tuple[int, int, str]:
    """Parse the boxed response format: reasoning followed by \\boxed{L,E}"""
    try:
        # Look for \boxed{L,E} pattern
        match = re.search(r'\\boxed\{(\d+)\s*,\s*(\d+)\}', response_text)
        if match:
            likeability = max(1, min(10, int(match.group(1))))
            emotional = max(1, min(10, int(match.group(2))))
            # Extract reasoning (everything before the boxed answer)
            reasoning = response_text[:match.start()].strip()
            # Clean up and truncate reasoning
            reasoning = re.sub(r'\s+', ' ', reasoning)[:300]
            return likeability, emotional, reasoning
    except (ValueError, AttributeError):
        pass
    return 0, 0, "[Parse error]"


class AsyncPersonaSimulator:
    """Async simulator for high throughput."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", 
                 model: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_concurrent: int = 32):
        self.base_url = base_url
        self.model = model
        self.max_concurrent = max_concurrent
        self.dataset = None
        self.semaphore = None
    
    def load_personas(self):
        """Load the personas dataset."""
        print("Loading nvidia/Nemotron-Personas-USA dataset...")
        self.dataset = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
        print(f"Loaded {len(self.dataset):,} personas")
    
    async def rate_single(self, session: aiohttp.ClientSession, 
                          idx: int, persona: dict, content: str) -> Optional[PersonaRating]:
        """Rate content as a single persona. Returns None on failure."""
        persona_desc = build_persona_prompt(persona)
        prompt = build_rating_prompt(persona_desc, content)
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,  # Allow room for reasoning + boxed answer
            "temperature": 0.7,
        }
        
        start = time.time()
        
        async with self.semaphore:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                ) as resp:
                    data = await resp.json()
                response_text = data['choices'][0]['message']['content']
            except Exception as e:
                # Return None on connection/request errors - will be retried on resume
                return None
        
        latency_ms = (time.time() - start) * 1000
        likeability, emotional, reasoning = parse_rating_response(response_text)
        
        # Reject parse errors (returns 0,0) - will be retried on resume
        if likeability == 0 or emotional == 0:
            return None
        
        return PersonaRating(
            uuid=persona.get('uuid', ''),
            idx=idx,
            likeability=likeability,
            emotional_activation=emotional,
            reasoning=reasoning,
            sex=persona.get('sex', ''),
            age=persona.get('age', 0),
            marital_status=persona.get('marital_status', ''),
            education_level=persona.get('education_level', ''),
            bachelors_field=persona.get('bachelors_field', ''),
            occupation=persona.get('occupation', ''),
            city=persona.get('city', ''),
            state=persona.get('state', ''),
            zipcode=persona.get('zipcode', ''),
            latency_ms=latency_ms,
        )
    
    async def simulate_batch(self, content: str, indices: list[int], 
                             progress_callback=None) -> tuple[list[PersonaRating], int]:
        """Run simulation on a batch of persona indices. Returns (successful_ratings, failure_count)."""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Allow lots of connections - vLLM handles batching internally
        connector = aiohttp.TCPConnector(limit=self.max_concurrent * 2, limit_per_host=self.max_concurrent * 2)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for idx in indices:
                persona = self.dataset[idx]
                task = self.rate_single(session, idx, persona, content)
                tasks.append(task)
            
            ratings = []
            failures = 0
            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
                try:
                    rating = await coro
                    if rating is not None:
                        ratings.append(rating)
                        if progress_callback:
                            progress_callback(rating)
                    else:
                        failures += 1
                except Exception as e:
                    failures += 1
        
        return ratings, failures


def load_checkpoint(output_dir: Path) -> set[int]:
    """Load completed indices from checkpoint."""
    completed = set()
    checkpoint_file = output_dir / "checkpoint.jsonl"
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                data = json.loads(line)
                completed.add(data['idx'])
    return completed


def save_checkpoint(output_dir: Path, ratings: list[PersonaRating]):
    """Append ratings to checkpoint file."""
    checkpoint_file = output_dir / "checkpoint.jsonl"
    with open(checkpoint_file, 'a') as f:
        for rating in ratings:
            f.write(json.dumps(asdict(rating)) + '\n')


def compute_stats(ratings: list[PersonaRating]) -> dict:
    """Compute aggregate statistics."""
    if not ratings:
        return {}
    
    likeability = [r.likeability for r in ratings]
    emotional = [r.emotional_activation for r in ratings]
    
    def stats(values):
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return {
            'mean': round(mean, 3),
            'std': round(variance ** 0.5, 3),
            'min': min(values),
            'max': max(values),
            'distribution': {i: values.count(i) for i in range(1, 11)}  # 1-10 scale
        }
    
    return {
        'n': len(ratings),
        'likeability': stats(likeability),
        'emotional_activation': stats(emotional),
    }


async def main():
    parser = argparse.ArgumentParser(description="Batch persona simulation")
    parser.add_argument("--content", type=str, required=True, help="Content to evaluate")
    parser.add_argument("--output", type=str, default="batch_run", help="Output directory")
    parser.add_argument("--num-personas", type=int, default=None, help="Number of personas (default: all)")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index")
    parser.add_argument("--batch-size", type=int, default=5000, help="Checkpoint every N personas")
    parser.add_argument("--max-concurrent", type=int, default=256, help="Max concurrent requests (crank this up for more GPUs)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Content: {args.content[:80]}{'...' if len(args.content) > 80 else ''}")
    print(f"Output: {output_dir}")
    print(f"Max concurrent: {args.max_concurrent}")
    print()
    
    simulator = AsyncPersonaSimulator(
        base_url=args.base_url,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )
    simulator.load_personas()
    
    # Determine which indices to process
    total_personas = len(simulator.dataset)
    end_idx = total_personas if args.num_personas is None else min(args.start_idx + args.num_personas, total_personas)
    all_indices = set(range(args.start_idx, end_idx))
    
    # Resume from checkpoint
    completed = set()
    if args.resume:
        completed = load_checkpoint(output_dir)
        print(f"Resuming: {len(completed):,} already completed")
    
    remaining = sorted(all_indices - completed)
    print(f"To process: {len(remaining):,} personas")
    
    if not remaining:
        print("Nothing to do!")
        return
    
    # Process in batches
    start_time = time.time()
    total_succeeded = 0
    total_failed = 0
    
    for batch_start in range(0, len(remaining), args.batch_size):
        batch_indices = remaining[batch_start:batch_start + args.batch_size]
        
        print(f"\nBatch {batch_start//args.batch_size + 1}: {len(batch_indices)} personas")
        
        ratings, failures = await simulator.simulate_batch(args.content, batch_indices)
        
        # Only save successful ratings - failed ones will be retried on resume
        if ratings:
            save_checkpoint(output_dir, ratings)
        total_succeeded += len(ratings)
        total_failed += failures
        
        # Print batch stats
        if ratings:
            stats = compute_stats(ratings)
            elapsed = time.time() - start_time
            rate = total_succeeded / elapsed if elapsed > 0 else 0
            remaining_count = len(remaining) - (total_succeeded + total_failed)
            eta = remaining_count / rate if rate > 0 else 0
            
            print(f"  Likeability: {stats['likeability']['mean']:.2f} ± {stats['likeability']['std']:.2f}")
            print(f"  Emotional:   {stats['emotional_activation']['mean']:.2f} ± {stats['emotional_activation']['std']:.2f}")
            print(f"  Succeeded: {len(ratings)}, Failed: {failures} (failures will retry on --resume)")
            print(f"  Progress: {total_succeeded:,} done ({rate:.1f}/s, ETA: {eta/3600:.1f}h)")
        else:
            print(f"  All {failures} requests failed! Is the vLLM server running?")
    
    # Final summary
    print(f"\n{'='*60}")
    print("BATCH COMPLETE")
    print('='*60)
    
    # Load all results from checkpoint
    checkpoint_file = output_dir / "checkpoint.jsonl"
    all_ratings = []
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            for line in f:
                data = json.loads(line)
                all_ratings.append(PersonaRating(**data))
    
    if all_ratings:
        overall_stats = compute_stats(all_ratings)
        
        total_time = time.time() - start_time
        print(f"Successful: {len(all_ratings):,} personas in {total_time/3600:.2f}h")
        print(f"Failed: {total_failed:,} (will retry on --resume)")
        print(f"Rate: {len(all_ratings)/total_time:.1f} personas/sec")
        print(f"\nLikeability:  {overall_stats['likeability']['mean']:.2f} ± {overall_stats['likeability']['std']:.2f}")
        print(f"Emotional:    {overall_stats['emotional_activation']['mean']:.2f} ± {overall_stats['emotional_activation']['std']:.2f}")
        
        # Save final summary
        summary = {
            'config': config,
            'total_succeeded': len(all_ratings),
            'total_failed_this_run': total_failed,
            'total_time_hours': total_time / 3600,
            'personas_per_second': len(all_ratings) / total_time if total_time > 0 else 0,
            'overall_stats': overall_stats,
        }
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("No successful ratings! Check if vLLM server is running.")


if __name__ == "__main__":
    asyncio.run(main())
