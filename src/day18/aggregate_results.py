"""
Aggregate simulation results for frontend visualization.

Takes checkpoint.jsonl and produces:
- by_zipcode.json: ratings aggregated by zipcode (for map)
- by_state.json: ratings by state
- by_demographics.json: breakdowns by sex, age, education, etc.
- overall.json: overall statistics

Usage:
    uv run python aggregate_results.py --input batch_run --output viz_data
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Rating:
    uuid: str
    idx: int
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


def load_ratings(input_path: Path) -> list[Rating]:
    """Load ratings from checkpoint file."""
    ratings = []
    checkpoint_file = input_path / "checkpoint.jsonl"
    
    print(f"Loading from {checkpoint_file}...")
    with open(checkpoint_file) as f:
        for line in f:
            data = json.loads(line)
            ratings.append(Rating(**data))
    
    print(f"Loaded {len(ratings):,} ratings")
    return ratings


def compute_group_stats(ratings: list[Rating]) -> dict:
    """Compute statistics for a group of ratings."""
    if not ratings:
        return None
    
    n = len(ratings)
    likeability = [r.likeability for r in ratings]
    emotional = [r.emotional_activation for r in ratings]
    
    like_mean = sum(likeability) / n
    like_std = (sum((x - like_mean) ** 2 for x in likeability) / n) ** 0.5
    
    emo_mean = sum(emotional) / n
    emo_std = (sum((x - emo_mean) ** 2 for x in emotional) / n) ** 0.5
    
    # Sample values for violin plots (keep file sizes reasonable)
    # Use all values if < 1000, otherwise sample
    max_samples = 1000
    likeability_sample = sorted(likeability) if len(likeability) <= max_samples else sorted(likeability)[::len(likeability)//max_samples]
    emotional_sample = sorted(emotional) if len(emotional) <= max_samples else sorted(emotional)[::len(emotional)//max_samples]
    
    return {
        'n': n,
        'likeability_mean': round(like_mean, 3),
        'likeability_std': round(like_std, 3),
        'likeability_min': min(likeability),
        'likeability_max': max(likeability),
        'likeability_dist': {i: likeability.count(i) for i in range(1, 11)},  # 1-10 scale
        'likeability_values': likeability_sample,
        'emotional_mean': round(emo_mean, 3),
        'emotional_std': round(emo_std, 3),
        'emotional_min': min(emotional),
        'emotional_max': max(emotional),
        'emotional_dist': {i: emotional.count(i) for i in range(1, 11)},  # 1-10 scale
        'emotional_values': emotional_sample,
    }


def aggregate_by_key(ratings: list[Rating], key_fn, min_count: int = 5) -> dict:
    """Aggregate ratings by a key function."""
    groups = defaultdict(list)
    
    for r in ratings:
        key = key_fn(r)
        if key:  # Skip empty keys
            groups[key].append(r)
    
    return {
        key: compute_group_stats(group)
        for key, group in groups.items()
        if len(group) >= min_count
    }


def get_age_bracket(age: int) -> str:
    """Convert age to bracket."""
    if age < 18:
        return 'under_18'
    elif age < 25:
        return '18-24'
    elif age < 35:
        return '25-34'
    elif age < 45:
        return '35-44'
    elif age < 55:
        return '45-54'
    elif age < 65:
        return '55-64'
    else:
        return '65+'


def main():
    parser = argparse.ArgumentParser(description="Aggregate simulation results")
    parser.add_argument("--input", type=str, required=True, help="Input directory with checkpoint.jsonl")
    parser.add_argument("--output", type=str, default="viz_data", help="Output directory for aggregated data")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum samples per group")
    parser.add_argument("--include-raw", action="store_true", help="Include raw ratings for interactive filtering (larger file)")
    parser.add_argument("--raw-sample", type=int, default=50000, help="Max raw ratings to include (for file size)")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Load ratings
    ratings = load_ratings(input_path)
    
    # Load config
    config_file = input_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)
    else:
        config = {}
    
    print("\nAggregating...")
    
    # Overall stats
    overall = compute_group_stats(ratings)
    overall['config'] = config
    
    with open(output_path / "overall.json", 'w') as f:
        json.dump(overall, f, indent=2)
    print(f"  Overall: n={overall['n']}, like={overall['likeability_mean']:.2f}, emo={overall['emotional_mean']:.2f}")
    
    # By zipcode (for map)
    by_zipcode = aggregate_by_key(ratings, lambda r: r.zipcode, min_count=args.min_count)
    with open(output_path / "by_zipcode.json", 'w') as f:
        json.dump(by_zipcode, f)
    print(f"  By zipcode: {len(by_zipcode)} groups")
    
    # By state
    by_state = aggregate_by_key(ratings, lambda r: r.state, min_count=args.min_count)
    with open(output_path / "by_state.json", 'w') as f:
        json.dump(by_state, f, indent=2)
    print(f"  By state: {len(by_state)} groups")
    
    # Demographics
    demographics = {
        'by_sex': aggregate_by_key(ratings, lambda r: r.sex, min_count=args.min_count),
        'by_age_bracket': aggregate_by_key(ratings, lambda r: get_age_bracket(r.age), min_count=args.min_count),
        'by_marital_status': aggregate_by_key(ratings, lambda r: r.marital_status, min_count=args.min_count),
        'by_education': aggregate_by_key(ratings, lambda r: r.education_level, min_count=args.min_count),
        'by_occupation': aggregate_by_key(ratings, lambda r: r.occupation if r.occupation not in ('no_occupation', 'not_in_workforce', '') else None, min_count=args.min_count),
    }
    
    with open(output_path / "by_demographics.json", 'w') as f:
        json.dump(demographics, f, indent=2)
    
    for demo_name, demo_data in demographics.items():
        print(f"  {demo_name}: {len(demo_data)} groups")
    
    # Sample reasoning for UI
    sample_ratings = []
    for r in ratings[:100]:  # First 100 as samples
        sample_ratings.append({
            'uuid': r.uuid,
            'likeability': r.likeability,
            'emotional_activation': r.emotional_activation,
            'reasoning': r.reasoning,
            'sex': r.sex,
            'age': r.age,
            'state': r.state,
            'occupation': r.occupation,
        })
    
    with open(output_path / "sample_ratings.json", 'w') as f:
        json.dump(sample_ratings, f, indent=2)
    print(f"  Sample ratings: {len(sample_ratings)}")
    
    # Raw ratings for interactive filtering (optional, can be large)
    if args.include_raw:
        import random
        raw_sample = ratings if len(ratings) <= args.raw_sample else random.sample(ratings, args.raw_sample)
        raw_ratings = [{
            'likeability': r.likeability,
            'emotional_activation': r.emotional_activation,
            'reasoning': r.reasoning,
            'sex': r.sex,
            'age': r.age,
            'marital_status': r.marital_status,
            'education_level': r.education_level,
            'occupation': r.occupation,
            'state': r.state,
        } for r in raw_sample]
        
        with open(output_path / "raw_ratings.json", 'w') as f:
            json.dump(raw_ratings, f)
        print(f"  Raw ratings: {len(raw_ratings)} (for interactive filtering)")
    
    print(f"\nSaved to {output_path}/")


if __name__ == "__main__":
    main()
