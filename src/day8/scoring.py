"""
Personality scoring for training and evaluation.

Training: Compute reward for a single question given target personality.
Evaluation: Compute full OCEAN scores from all answers.
"""

from dataclasses import dataclass, field
from typing import Callable

from data import QuestionBank, Question, OCEAN_NAMES, FACET_NAMES


@dataclass
class TargetPersonality:
    """
    Target personality profile for training.
    
    Scores are on 1-5 scale (matching answer scale):
    - 1 = Very low on this dimension
    - 3 = Neutral/average
    - 5 = Very high on this dimension
    
    Can specify at OCEAN level (coarse) or facet level (fine-grained).
    """
    # OCEAN-level targets (used if facet-level not specified)
    neuroticism: float = 3.0
    extraversion: float = 3.0
    openness: float = 3.0
    agreeableness: float = 3.0
    conscientiousness: float = 3.0
    
    # Optional: facet-level overrides (dict of facet_name -> target)
    facet_targets: dict[str, float] = field(default_factory=dict)
    
    def get_target_for_question(self, q: Question) -> float:
        """Get the target score for a specific question."""
        # Check facet-level override first
        if q.facet_name in self.facet_targets:
            return self.facet_targets[q.facet_name]
        
        # Fall back to OCEAN-level
        ocean_targets = {
            "N": self.neuroticism,
            "E": self.extraversion,
            "O": self.openness,
            "A": self.agreeableness,
            "C": self.conscientiousness,
        }
        return ocean_targets[q.ocean]
    
    @classmethod
    def high_agreeableness(cls) -> "TargetPersonality":
        """People-pleasing personality (what LLMs default to)."""
        return cls(agreeableness=5.0, neuroticism=1.0, extraversion=4.0)
    
    @classmethod
    def low_agreeableness(cls) -> "TargetPersonality":
        """Disagreeable / contrarian personality."""
        return cls(agreeableness=1.0, neuroticism=3.0, extraversion=3.0)
    
    @classmethod
    def introvert(cls) -> "TargetPersonality":
        """Introverted personality."""
        return cls(extraversion=1.0)
    
    @classmethod
    def neurotic(cls) -> "TargetPersonality":
        """High neuroticism personality."""
        return cls(neuroticism=5.0)
    
    @classmethod
    def stable(cls) -> "TargetPersonality":
        """Emotionally stable personality."""
        return cls(neuroticism=1.0)


class PersonalityScorer:
    """
    Handles both training rewards and full personality evaluation.
    """
    
    def __init__(self, question_bank: QuestionBank):
        self.bank = question_bank
    
    # ==================== TRAINING ====================
    
    def compute_reward(
        self, 
        question: Question, 
        answer: int,
        target: TargetPersonality,
        reward_type: str = "negative_l1",
    ) -> float:
        """
        Compute reward for a single question-answer pair.
        
        Args:
            question: The question that was asked
            answer: Model's answer (1-5 scale)
            target: Target personality to optimize toward
            reward_type: How to compute reward
                - "negative_l1": -|answer - target| (range: -4 to 0)
                - "negative_l2": -(answer - target)^2 (range: -16 to 0)
                - "match": 1.0 if rounded answer matches, else 0.0
        
        Returns:
            Scalar reward (higher is better)
        """
        # Get the target answer for this question
        target_answer = target.get_target_for_question(question)
        
        # Handle reverse scoring: if question is reversed, we need to flip our target
        # because a "high agreeableness" answer to a reversed agreeableness question
        # would actually be a LOW numerical answer
        if question.is_reversed:
            target_answer = 6 - target_answer
        
        if reward_type == "negative_l1":
            return -abs(answer - target_answer)
        elif reward_type == "negative_l2":
            return -((answer - target_answer) ** 2)
        elif reward_type == "match":
            return 1.0 if round(answer) == round(target_answer) else 0.0
        else:
            raise ValueError(f"Unknown reward type: {reward_type}")
    
    def compute_reward_from_probs(
        self,
        question: Question,
        answer_probs: list[float],  # [p(1), p(2), p(3), p(4), p(5)]
        target: TargetPersonality,
    ) -> float:
        """
        Compute expected reward given probability distribution over answers.
        Useful for differentiable training.
        
        Args:
            question: The question asked
            answer_probs: Probability of each answer (1-5), should sum to 1
            target: Target personality
        
        Returns:
            Expected reward (sum of p(answer) * reward(answer))
        """
        assert len(answer_probs) == 5
        expected_reward = 0.0
        for i, prob in enumerate(answer_probs):
            answer = i + 1  # Convert to 1-5 scale
            reward = self.compute_reward(question, answer, target, "negative_l1")
            expected_reward += prob * reward
        return expected_reward
    
    # ==================== EVALUATION ====================
    
    def compute_personality(
        self,
        answers: dict[int, int],  # question_id -> answer (1-5)
        normalize: bool = True,
    ) -> dict:
        """
        Compute full personality profile from answers.
        
        Args:
            answers: Dict mapping question_id to answer (1-5)
            normalize: If True, normalize scores to 1-5 scale
        
        Returns:
            Dict with OCEAN scores and facet breakdowns
        """
        # Accumulate scores per facet (30 facets)
        facet_scores = {i: [] for i in range(30)}
        
        for qid, answer in answers.items():
            q = self.bank[qid]
            scored = q.score_answer(answer)  # Handles reverse scoring
            facet_scores[q.facet_idx].append(scored)
        
        # Average each facet
        facet_means = {}
        for facet_idx, scores in facet_scores.items():
            if scores:
                facet_means[facet_idx] = sum(scores) / len(scores)
            else:
                facet_means[facet_idx] = 3.0  # Default to neutral
        
        # Compute OCEAN scores (average of their 6 facets each)
        ocean_scores = {}
        ocean_facets = {}
        
        for ocean_idx, ocean in enumerate(OCEAN_NAMES):
            # Get the 6 facet indices for this ocean dimension
            facet_indices = [ocean_idx + 5 * i for i in range(6)]
            facet_values = [facet_means[fi] for fi in facet_indices]
            
            ocean_scores[ocean] = sum(facet_values) / len(facet_values)
            ocean_facets[ocean] = {
                FACET_NAMES[ocean][i]: facet_values[i] 
                for i in range(6)
            }
        
        return {
            "ocean": ocean_scores,
            "facets": ocean_facets,
            "n_questions": len(answers),
        }
    
    def evaluate_test_set(
        self,
        get_answer: Callable[[Question], int],
    ) -> dict:
        """
        Run model through all test questions and compute personality.
        
        Args:
            get_answer: Function that takes a Question and returns an answer (1-5)
        
        Returns:
            Personality profile from test set answers
        """
        answers = {}
        for qid in self.bank.test_ids:
            q = self.bank[qid]
            answer = get_answer(q)
            answers[qid] = answer
        
        return self.compute_personality(answers)
    
    def evaluate_train_set(
        self,
        get_answer: Callable[[Question], int],
    ) -> dict:
        """Same as evaluate_test_set but on training questions."""
        answers = {}
        for qid in self.bank.train_ids:
            q = self.bank[qid]
            answer = get_answer(q)
            answers[qid] = answer
        
        return self.compute_personality(answers)


# Quick test / demo
if __name__ == "__main__":
    bank = QuestionBank()
    scorer = PersonalityScorer(bank)
    
    # Define a target personality
    target = TargetPersonality.low_agreeableness()
    print(f"Target: A={target.agreeableness}, N={target.neuroticism}")
    
    # Sample a question and compute rewards for different answers
    q = bank.sample_train_question()
    print(f"\nQuestion: {q.text}")
    print(f"  Ocean: {q.ocean}, Facet: {q.facet_name}, Reversed: {q.is_reversed}")
    print(f"  Target for this Q: {target.get_target_for_question(q)}")
    
    print("\nRewards for different answers:")
    for ans in [1, 2, 3, 4, 5]:
        r = scorer.compute_reward(q, ans, target)
        print(f"  Answer {ans}: reward = {r:.2f}")
    
    # Mock evaluation: answer everything as 3
    def mock_answer(q: Question) -> int:
        return 3
    
    result = scorer.evaluate_test_set(mock_answer)
    print(f"\nMock evaluation (all 3s):")
    print(f"  OCEAN: {result['ocean']}")

