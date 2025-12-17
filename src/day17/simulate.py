"""
Persona Simulation Backend

Simulates how a population of synthetic personas would react to content.
Uses vLLM-served model with OpenAI-compatible API.

Usage:
    # Start vLLM server first:
    vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
    
    # Run simulation:
    uv run python simulate.py --content "Your tweet or micro-blog here" --num-personas 100
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import re

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


@dataclass
class PersonaRating:
    """Rating from a single persona."""
    uuid: str
    likeability: int  # 1-5
    emotional_activation: int  # 1-5
    reasoning: str
    
    # Demographics for analysis
    sex: str
    age: int
    marital_status: str
    education_level: str
    bachelors_field: str
    occupation: str
    city: str
    state: str
    zipcode: str
    
    # Timing
    latency_ms: float


def build_persona_prompt(persona: dict) -> str:
    """Build a rich persona description from all fields."""
    parts = []
    
    # Core identity
    parts.append(f"You are {persona.get('persona', 'a person')}.")
    
    # Cultural background
    if persona.get('cultural_background'):
        parts.append(f"\nBackground: {persona['cultural_background']}")
    
    # Professional
    if persona.get('professional_persona'):
        parts.append(f"\nProfessional life: {persona['professional_persona']}")
    
    # Interests
    if persona.get('hobbies_and_interests'):
        parts.append(f"\nHobbies & interests: {persona['hobbies_and_interests']}")
    
    # Arts/culture
    if persona.get('arts_persona'):
        parts.append(f"\nArts & culture: {persona['arts_persona']}")
    
    # Sports
    if persona.get('sports_persona'):
        parts.append(f"\nSports & fitness: {persona['sports_persona']}")
    
    # Culinary
    if persona.get('culinary_persona'):
        parts.append(f"\nFood & cooking: {persona['culinary_persona']}")
    
    # Travel
    if persona.get('travel_persona'):
        parts.append(f"\nTravel: {persona['travel_persona']}")
    
    # Skills
    if persona.get('skills_and_expertise'):
        parts.append(f"\nSkills: {persona['skills_and_expertise']}")
    
    # Goals
    if persona.get('career_goals_and_ambitions'):
        parts.append(f"\nGoals: {persona['career_goals_and_ambitions']}")
    
    # Demographics
    demo_parts = []
    if persona.get('age'):
        demo_parts.append(f"{persona['age']} years old")
    if persona.get('sex'):
        demo_parts.append(persona['sex'].lower())
    if persona.get('marital_status'):
        demo_parts.append(persona['marital_status'].replace('_', ' '))
    if persona.get('education_level'):
        demo_parts.append(f"education: {persona['education_level'].replace('_', ' ')}")
    if persona.get('occupation') and persona['occupation'] != 'no_occupation':
        demo_parts.append(f"works as: {persona['occupation'].replace('_', ' ')}")
    if persona.get('city') and persona.get('state'):
        demo_parts.append(f"lives in {persona['city']}, {persona['state']}")
    
    if demo_parts:
        parts.append(f"\nDemographics: {', '.join(demo_parts)}")
    
    return '\n'.join(parts)


def build_rating_prompt(persona_description: str, content: str) -> str:
    """Build the full prompt for rating content."""
    return f"""{persona_description}

---

You are browsing social media and see this post:

\"\"\"{content}\"\"\"

As this persona, rate your reaction:

1. LIKEABILITY: How much do you like or dislike this post?
   1 = Strongly dislike
   2 = Somewhat dislike  
   3 = Neutral
   4 = Somewhat like
   5 = Strongly like

2. EMOTIONAL ACTIVATION: How emotionally stirred does this make you feel? (This could be positive OR negative emotion)
   1 = Completely indifferent
   2 = Slightly engaged
   3 = Moderately engaged
   4 = Quite stirred up
   5 = Very emotionally activated

Respond in this exact JSON format:
{{"likeability": <1-5>, "emotional_activation": <1-5>, "reasoning": "<one sentence explaining your reaction as this persona>"}}"""


def parse_rating_response(response_text: str) -> tuple[int, int, str]:
    """Parse the JSON response from the model."""
    # Try to find JSON in the response
    try:
        # Look for JSON pattern
        match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            likeability = int(data.get('likeability', 3))
            emotional = int(data.get('emotional_activation', 3))
            reasoning = str(data.get('reasoning', ''))
            
            # Clamp to valid range
            likeability = max(1, min(5, likeability))
            emotional = max(1, min(5, emotional))
            
            return likeability, emotional, reasoning
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    
    # Fallback: return neutral
    return 3, 3, f"[Parse error] {response_text[:100]}"


class PersonaSimulator:
    """Simulates persona reactions to content."""
    
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.dataset = None
    
    def load_personas(self):
        """Load the personas dataset."""
        print("Loading nvidia/Nemotron-Personas-USA dataset...")
        self.dataset = load_dataset("nvidia/Nemotron-Personas-USA", split="train")
        print(f"Loaded {len(self.dataset):,} personas")
    
    def rate_content(self, persona: dict, content: str) -> PersonaRating:
        """Have a persona rate a piece of content."""
        persona_desc = build_persona_prompt(persona)
        prompt = build_rating_prompt(persona_desc, content)
        
        start = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        
        latency_ms = (time.time() - start) * 1000
        
        response_text = response.choices[0].message.content
        likeability, emotional, reasoning = parse_rating_response(response_text)
        
        return PersonaRating(
            uuid=persona.get('uuid', ''),
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
    
    def simulate(self, content: str, num_personas: Optional[int] = None, 
                 start_idx: int = 0, batch_size: int = 1) -> list[PersonaRating]:
        """Run simulation across personas."""
        if self.dataset is None:
            self.load_personas()
        
        if num_personas is None:
            num_personas = len(self.dataset)
        
        end_idx = min(start_idx + num_personas, len(self.dataset))
        
        ratings = []
        pbar = tqdm(range(start_idx, end_idx), desc="Simulating")
        
        for i in pbar:
            persona = self.dataset[i]
            rating = self.rate_content(persona, content)
            ratings.append(rating)
            
            # Update progress bar with running averages
            if ratings:
                avg_like = sum(r.likeability for r in ratings) / len(ratings)
                avg_emo = sum(r.emotional_activation for r in ratings) / len(ratings)
                pbar.set_postfix({
                    'like': f'{avg_like:.2f}',
                    'emo': f'{avg_emo:.2f}',
                    'ms': f'{rating.latency_ms:.0f}'
                })
        
        return ratings


def compute_stats(ratings: list[PersonaRating]) -> dict:
    """Compute aggregate statistics from ratings."""
    if not ratings:
        return {}
    
    likeability = [r.likeability for r in ratings]
    emotional = [r.emotional_activation for r in ratings]
    
    def stats(values):
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / n
        return {
            'mean': mean,
            'std': variance ** 0.5,
            'min': min(values),
            'max': max(values),
            'distribution': {i: values.count(i) for i in range(1, 6)}
        }
    
    return {
        'n': len(ratings),
        'likeability': stats(likeability),
        'emotional_activation': stats(emotional),
    }


def compute_demographic_breakdown(ratings: list[PersonaRating]) -> dict:
    """Break down ratings by demographic groups."""
    breakdowns = {}
    
    # Group by various demographics
    demographics = ['sex', 'marital_status', 'education_level', 'state']
    
    for demo in demographics:
        groups = {}
        for r in ratings:
            key = getattr(r, demo) or 'unknown'
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        breakdowns[demo] = {
            key: compute_stats(group_ratings)
            for key, group_ratings in groups.items()
            if len(group_ratings) >= 5  # Only groups with enough data
        }
    
    # Age brackets
    age_brackets = {'18-24': [], '25-34': [], '35-44': [], '45-54': [], '55-64': [], '65+': []}
    for r in ratings:
        age = r.age
        if age < 25:
            age_brackets['18-24'].append(r)
        elif age < 35:
            age_brackets['25-34'].append(r)
        elif age < 45:
            age_brackets['35-44'].append(r)
        elif age < 55:
            age_brackets['45-54'].append(r)
        elif age < 65:
            age_brackets['55-64'].append(r)
        else:
            age_brackets['65+'].append(r)
    
    breakdowns['age_bracket'] = {
        key: compute_stats(group_ratings)
        for key, group_ratings in age_brackets.items()
        if len(group_ratings) >= 5
    }
    
    return breakdowns


def main():
    parser = argparse.ArgumentParser(description="Simulate persona reactions to content")
    parser.add_argument("--content", type=str, required=True, help="Content to evaluate")
    parser.add_argument("--num-personas", type=int, default=100, help="Number of personas to simulate")
    parser.add_argument("--start-idx", type=int, default=0, help="Starting persona index")
    parser.add_argument("--output", type=str, default="simulation_results.json", help="Output file")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    args = parser.parse_args()
    
    print(f"Content: {args.content[:100]}{'...' if len(args.content) > 100 else ''}")
    print(f"Personas: {args.num_personas} starting at {args.start_idx}")
    print(f"Server: {args.base_url}")
    print()
    
    simulator = PersonaSimulator(base_url=args.base_url, model=args.model)
    
    start_time = time.time()
    ratings = simulator.simulate(
        content=args.content,
        num_personas=args.num_personas,
        start_idx=args.start_idx,
    )
    total_time = time.time() - start_time
    
    # Compute statistics
    overall_stats = compute_stats(ratings)
    demographic_breakdown = compute_demographic_breakdown(ratings)
    
    # Build results
    results = {
        'config': {
            'content': args.content,
            'num_personas': args.num_personas,
            'start_idx': args.start_idx,
            'model': args.model,
        },
        'timing': {
            'total_seconds': total_time,
            'personas_per_second': len(ratings) / total_time if total_time > 0 else 0,
            'avg_latency_ms': sum(r.latency_ms for r in ratings) / len(ratings) if ratings else 0,
        },
        'overall': overall_stats,
        'by_demographic': demographic_breakdown,
        'ratings': [asdict(r) for r in ratings],
    }
    
    # Save
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print('='*60)
    print(f"Total personas: {len(ratings)}")
    print(f"Total time: {total_time:.1f}s ({len(ratings)/total_time:.1f} personas/sec)")
    print(f"\nLikeability:  {overall_stats['likeability']['mean']:.2f} ± {overall_stats['likeability']['std']:.2f}")
    print(f"Emotional:    {overall_stats['emotional_activation']['mean']:.2f} ± {overall_stats['emotional_activation']['std']:.2f}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
