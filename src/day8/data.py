"""
Question bank and train/test split for Big Five personality training.
Uses IPIP-NEO-300 format: 30 facets × 10 questions each.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


# The 5 OCEAN dimensions
OCEAN_NAMES = ["N", "E", "O", "A", "C"]
OCEAN_FULL_NAMES = {
    "N": "Neuroticism",
    "E": "Extraversion", 
    "O": "Openness",
    "A": "Agreeableness",
    "C": "Conscientiousness",
}

# 6 facets per OCEAN dimension (30 total)
FACET_NAMES = {
    "N": ["anxiety", "anger", "depression", "self_consciousness", "immoderation", "vulnerability"],
    "E": ["friendliness", "gregariousness", "assertiveness", "activity_level", "excitement_seeking", "cheerfulness"],
    "O": ["imagination", "artistic_interests", "emotionality", "adventurousness", "intellect", "liberalism"],
    "A": ["trust", "morality", "altruism", "cooperation", "modesty", "sympathy"],
    "C": ["self_efficacy", "orderliness", "dutifulness", "achievement_striving", "self_discipline", "cautiousness"],
}

# Reverse-scored items for IPIP-300 (from five-factor-e reference)
REVERSED_ITEMS_300 = {
    69, 99, 109, 118, 120, 129, 138, 139, 144, 148, 149, 150, 151, 152, 156, 157,
    158, 159, 160, 162, 163, 164, 165, 167, 168, 169, 171, 173, 174, 175, 176, 178,
    179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195,
    196, 197, 198, 199, 201, 203, 204, 205, 206, 208, 209, 210, 211, 212, 213, 214,
    215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230,
    231, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264,
    265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
    281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
    297, 298, 299, 300,
}


@dataclass
class Question:
    """A single personality question."""
    id: int
    text: str
    facet_idx: int      # 0-29
    ocean_idx: int      # 0-4 (N, E, O, A, C)
    ocean: str          # "N", "E", "O", "A", "C"
    facet_name: str     # e.g., "anxiety", "friendliness"
    is_reversed: bool   # True if answer should be reverse-scored
    
    def score_answer(self, answer: int) -> int:
        """
        Convert raw answer (1-5) to scored value (1-5).
        Handles reverse scoring.
        """
        assert 1 <= answer <= 5, f"Answer must be 1-5, got {answer}"
        if self.is_reversed:
            return 6 - answer  # 1->5, 2->4, 3->3, 4->2, 5->1
        return answer


class QuestionBank:
    """
    Holds all 300 questions with train/test split.
    
    The interleaving pattern for IPIP-300:
    - Questions cycle through all 30 facets
    - Question (id-1) % 30 gives facet index (0-29)
    - Each facet gets 10 questions
    - Facets 0,5,10,15,20,25 → Neuroticism
    - Facets 1,6,11,16,21,26 → Extraversion
    - etc.
    """
    
    def __init__(
        self, 
        questions_path: str | Path | None = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ):
        """
        Load questions and create train/test split.
        
        Args:
            questions_path: Path to questions.json (defaults to bundled data)
            train_ratio: Fraction of questions per facet for training (default 0.8 = 8/10)
            seed: Random seed for split
        """
        if questions_path is None:
            # Default to bundled data
            questions_path = Path(__file__).parent / "data" / "questions.json"
        
        with open(questions_path) as f:
            data = json.load(f)
        
        self.questions: dict[int, Question] = {}
        self._build_questions(data["questions"])
        
        # Create balanced train/test split (per facet)
        self.train_ids: list[int] = []
        self.test_ids: list[int] = []
        self._create_split(train_ratio, seed)
    
    def _build_questions(self, raw_questions: list[dict]):
        """Parse raw questions and compute facet/ocean mappings."""
        for q in raw_questions:
            qid = q["id"]
            # Facet index: questions cycle through 30 facets
            facet_idx = (qid - 1) % 30
            
            # Ocean index: facets are interleaved N,E,O,A,C,N,E,O,A,C...
            # Facets 0,5,10,15,20,25 → N (idx 0)
            # Facets 1,6,11,16,21,26 → E (idx 1)
            # etc.
            ocean_idx = facet_idx % 5
            ocean = OCEAN_NAMES[ocean_idx]
            
            # Which of the 6 facets within this ocean dimension?
            facet_within_ocean = facet_idx // 5  # 0-5
            facet_name = FACET_NAMES[ocean][facet_within_ocean]
            
            self.questions[qid] = Question(
                id=qid,
                text=q["text"],
                facet_idx=facet_idx,
                ocean_idx=ocean_idx,
                ocean=ocean,
                facet_name=facet_name,
                is_reversed=qid in REVERSED_ITEMS_300,
            )
    
    def _create_split(self, train_ratio: float, seed: int):
        """Create balanced train/test split within each facet."""
        rng = random.Random(seed)
        
        # Group questions by facet
        facet_questions: dict[int, list[int]] = {i: [] for i in range(30)}
        for qid, q in self.questions.items():
            facet_questions[q.facet_idx].append(qid)
        
        # Split each facet
        for facet_idx in range(30):
            qids = facet_questions[facet_idx]
            rng.shuffle(qids)
            n_train = int(len(qids) * train_ratio)
            self.train_ids.extend(qids[:n_train])
            self.test_ids.extend(qids[n_train:])
        
        # Sort for reproducibility
        self.train_ids.sort()
        self.test_ids.sort()
    
    def get_train_questions(self) -> list[Question]:
        """Get all training questions."""
        return [self.questions[qid] for qid in self.train_ids]
    
    def get_test_questions(self) -> list[Question]:
        """Get all test questions."""
        return [self.questions[qid] for qid in self.test_ids]
    
    def sample_train_question(self, rng: random.Random | None = None) -> Question:
        """Sample a random training question."""
        if rng is None:
            rng = random.Random()
        qid = rng.choice(self.train_ids)
        return self.questions[qid]
    
    def __getitem__(self, qid: int) -> Question:
        return self.questions[qid]
    
    def __len__(self) -> int:
        return len(self.questions)


# Quick test
if __name__ == "__main__":
    bank = QuestionBank()
    print(f"Total questions: {len(bank)}")
    print(f"Train: {len(bank.train_ids)}, Test: {len(bank.test_ids)}")
    
    # Show a few examples
    print("\nSample questions:")
    for qid in [1, 2, 3, 4, 5, 31, 32]:
        q = bank[qid]
        print(f"  Q{qid}: [{q.ocean}:{q.facet_name}] {q.text} (reversed={q.is_reversed})")
    
    # Verify facet distribution
    print("\nQuestions per facet in train set:")
    from collections import Counter
    train_facets = Counter(bank[qid].facet_idx for qid in bank.train_ids)
    print(f"  {dict(sorted(train_facets.items()))}")

