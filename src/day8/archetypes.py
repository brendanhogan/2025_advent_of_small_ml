"""
Predefined personality archetypes for training targets.
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from scoring import TargetPersonality


# Predefined archetypes with descriptive names
ARCHETYPES = {
    "default_llm": {
        "description": "Typical LLM personality - high agreeableness, low neuroticism, high extraversion",
        "neuroticism": 1.5,
        "extraversion": 4.5,
        "openness": 4.0,
        "agreeableness": 5.0,
        "conscientiousness": 4.0,
    },
    "balanced": {
        "description": "Neutral on all dimensions",
        "neuroticism": 3.0,
        "extraversion": 3.0,
        "openness": 3.0,
        "agreeableness": 3.0,
        "conscientiousness": 3.0,
    },
    
    # ============================================
    # EXTREME TRAINING TARGETS
    # Designed to be maximally different from default LLM
    # ============================================
    
    # The jerk: Extremely disagreeable, blunt, competitive, doesn't care what you think
    "jerk": {
        "description": "Maximally disagreeable: blunt, competitive, critical, zero people-pleasing",
        "neuroticism": 2.0,      # Confident, not anxious
        "extraversion": 4.0,     # Still outgoing (loud asshole, not quiet one)
        "openness": 3.0,         # Neutral
        "agreeableness": 1.0,    # EXTREME LOW - the point
        "conscientiousness": 3.0,
    },
    
    # The Neurotic Mess: Anxious, self-doubting, emotionally volatile
    "neurotic": {
        "description": "Maximally neurotic: anxious, insecure, overthinks everything, easily stressed",
        "neuroticism": 5.0,      # EXTREME HIGH - the point
        "extraversion": 2.0,     # Withdrawn due to anxiety
        "openness": 3.0,         # Neutral
        "agreeableness": 4.0,    # Still agreeable (anxious people-pleaser)
        "conscientiousness": 2.0, # Paralyzed by perfectionism/anxiety
    },
    
    # The Mad Artist: Wildly creative, unconventional, head in the clouds
    "creative_chaos": {
        "description": "Maximally open: wildly imaginative, unconventional, abstract thinker, hates routine",
        "neuroticism": 3.0,      # Neutral (can be moody but not defining)
        "extraversion": 3.0,     # Neutral
        "openness": 5.0,         # EXTREME HIGH - the point
        "agreeableness": 2.0,    # Artists can be difficult
        "conscientiousness": 1.0, # Chaotic, spontaneous, hates structure
    },
    
    # The Cold Logician: Disagreeable but highly analytical, Spock-like
    "cold_logician": {
        "description": "Disagreeable + high conscientiousness: brutally logical, no emotional hand-holding",
        "neuroticism": 1.0,      # Ice cold, unflappable
        "extraversion": 2.0,     # Reserved, doesn't need social validation
        "openness": 4.0,         # Intellectually curious
        "agreeableness": 1.0,    # EXTREME LOW - won't sugarcoat
        "conscientiousness": 5.0, # EXTREME HIGH - rigorous, precise
    },
}


def get_archetype(name: str) -> TargetPersonality:
    """Get a predefined archetype as a TargetPersonality."""
    if name not in ARCHETYPES:
        available = ", ".join(ARCHETYPES.keys())
        raise ValueError(f"Unknown archetype '{name}'. Available: {available}")
    
    arch = ARCHETYPES[name]
    return TargetPersonality(
        neuroticism=arch["neuroticism"],
        extraversion=arch["extraversion"],
        openness=arch["openness"],
        agreeableness=arch["agreeableness"],
        conscientiousness=arch["conscientiousness"],
    )


def load_target_from_json(path: str | Path) -> TargetPersonality:
    """
    Load target personality from JSON file.
    
    JSON format:
    {
        "neuroticism": 2.0,
        "extraversion": 4.0,
        "openness": 3.5,
        "agreeableness": 1.5,
        "conscientiousness": 3.0,
        "facet_targets": {  // optional
            "anxiety": 1.0,
            "assertiveness": 5.0
        }
    }
    
    OR just specify an archetype name:
    {
        "archetype": "contrarian"
    }
    """
    with open(path) as f:
        data = json.load(f)
    
    # Check if it's an archetype reference
    if "archetype" in data:
        return get_archetype(data["archetype"])
    
    return TargetPersonality(
        neuroticism=data.get("neuroticism", 3.0),
        extraversion=data.get("extraversion", 3.0),
        openness=data.get("openness", 3.0),
        agreeableness=data.get("agreeableness", 3.0),
        conscientiousness=data.get("conscientiousness", 3.0),
        facet_targets=data.get("facet_targets", {}),
    )


def save_archetype_examples():
    """Save example archetype JSONs for reference."""
    output_dir = Path(__file__).parent / "example_targets"
    output_dir.mkdir(exist_ok=True)
    
    for name, arch in ARCHETYPES.items():
        path = output_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(arch, f, indent=2)
    
    print(f"Saved {len(ARCHETYPES)} example targets to {output_dir}")


def list_archetypes() -> None:
    """Print available archetypes."""
    print("Available personality archetypes:")
    print("-" * 60)
    for name, arch in ARCHETYPES.items():
        print(f"\n{name}:")
        print(f"  {arch['description']}")
        print(f"  N={arch['neuroticism']:.1f} E={arch['extraversion']:.1f} "
              f"O={arch['openness']:.1f} A={arch['agreeableness']:.1f} C={arch['conscientiousness']:.1f}")


if __name__ == "__main__":
    list_archetypes()
    save_archetype_examples()

