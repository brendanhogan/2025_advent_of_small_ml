"""
Round-robin tournament with Gradio web UI for human preference feedback.

Uses Gradio with share=True to create a public URL accessible from anywhere,
which works seamlessly on SLURM clusters without port forwarding.
"""

import os
import time
import random
import threading
from itertools import combinations, product
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from PIL import Image

# Fix Gradio temp directory permissions on shared clusters
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/.cache/gradio")
os.makedirs(os.path.expanduser("~/.cache/gradio"), exist_ok=True)

import gradio as gr


@dataclass
class EvalTournamentState:
    """Tracks the state of a model vs GPT-4.1 evaluation tournament."""
    
    round_num: int
    model_image_paths: List[str]
    model_prompts: List[str]
    gpt_image_paths: List[str]
    gpt_prompts: List[str]
    # Each matchup is (model_idx, gpt_idx, is_model_on_left)
    matchups: List[Tuple[int, int, bool]] = field(default_factory=list)
    current_matchup_idx: int = 0
    model_wins: int = 0
    gpt_wins: int = 0
    # Detailed results: list of (model_idx, gpt_idx, winner: 'model' or 'gpt')
    results: List[Tuple[int, int, str]] = field(default_factory=list)
    completed: bool = False
    is_eval: bool = True
    
    def __post_init__(self):
        # Generate all 4x4 matchups with randomized left/right
        matchups = []
        for model_idx in range(len(self.model_image_paths)):
            for gpt_idx in range(len(self.gpt_image_paths)):
                is_model_on_left = random.random() < 0.5
                matchups.append((model_idx, gpt_idx, is_model_on_left))
        # Shuffle the order too
        random.shuffle(matchups)
        self.matchups = matchups
    
    def get_current_matchup(self) -> Optional[Tuple[int, int, bool]]:
        if self.current_matchup_idx >= len(self.matchups):
            return None
        return self.matchups[self.current_matchup_idx]
    
    def get_current_images(self) -> Tuple[str, str]:
        """Returns (left_path, right_path)."""
        matchup = self.get_current_matchup()
        if matchup is None:
            return None, None
        
        model_idx, gpt_idx, is_model_on_left = matchup
        model_path = self.model_image_paths[model_idx]
        gpt_path = self.gpt_image_paths[gpt_idx]
        
        if is_model_on_left:
            return model_path, gpt_path
        else:
            return gpt_path, model_path
    
    def record_vote(self, winner_side: str) -> bool:
        """Record a vote ('left' or 'right'). Returns True if tournament is complete."""
        matchup = self.get_current_matchup()
        if matchup is None:
            return True
        
        model_idx, gpt_idx, is_model_on_left = matchup
        
        # Determine who actually won
        if is_model_on_left:
            winner = 'model' if winner_side == 'left' else 'gpt'
        else:
            winner = 'gpt' if winner_side == 'left' else 'model'
        
        if winner == 'model':
            self.model_wins += 1
        else:
            self.gpt_wins += 1
        
        self.results.append((model_idx, gpt_idx, winner))
        self.current_matchup_idx += 1
        
        if self.current_matchup_idx >= len(self.matchups):
            self.completed = True
            return True
        return False
    
    def get_results_summary(self) -> Dict:
        """Get a summary of the evaluation results."""
        total = self.model_wins + self.gpt_wins
        return {
            'model_wins': self.model_wins,
            'gpt_wins': self.gpt_wins,
            'total_matchups': total,
            'model_win_rate': self.model_wins / total if total > 0 else 0,
            'gpt_win_rate': self.gpt_wins / total if total > 0 else 0,
            'detailed_results': self.results,
            'model_prompts': self.model_prompts,
            'gpt_prompts': self.gpt_prompts,
        }


@dataclass
class TournamentState:
    """Tracks the state of a round-robin tournament."""
    
    round_num: int
    image_paths: List[str]
    prompts: List[str]
    matchups: List[Tuple[int, int]] = field(default_factory=list)
    current_matchup_idx: int = 0
    votes: Dict[int, int] = field(default_factory=dict)  # image_idx -> win count
    completed: bool = False
    
    def __post_init__(self):
        # Generate all matchups (round-robin)
        n = len(self.image_paths)
        self.matchups = list(combinations(range(n), 2))
        # Initialize vote counts
        self.votes = {i: 0 for i in range(n)}
    
    def get_current_matchup(self) -> Optional[Tuple[int, int]]:
        if self.current_matchup_idx >= len(self.matchups):
            return None
        return self.matchups[self.current_matchup_idx]
    
    def record_vote(self, winner: str) -> bool:
        """Record a vote. Returns True if tournament is complete."""
        matchup = self.get_current_matchup()
        if matchup is None:
            return True
        
        left_idx, right_idx = matchup
        if winner == 'left':
            self.votes[left_idx] += 1
        else:
            self.votes[right_idx] += 1
        
        self.current_matchup_idx += 1
        
        if self.current_matchup_idx >= len(self.matchups):
            self.completed = True
            return True
        return False
    
    def get_win_rates(self) -> List[float]:
        """Get win rate for each image (wins / total_games)."""
        n = len(self.image_paths)
        games_per_image = n - 1  # Each image plays against every other image once
        return [self.votes[i] / games_per_image for i in range(n)]


class TournamentServer:
    """Gradio server for running the image tournament."""
    
    def __init__(self, share: bool = True):
        self.share = share
        self.state = None  # Can be TournamentState or EvalTournamentState
        self.demo: Optional[gr.Blocks] = None
        self._server_thread: Optional[threading.Thread] = None
        self.public_url: Optional[str] = None
    
    def _get_current_images(self):
        """Get the current pair of images to compare."""
        if self.state is None or self.state.completed:
            return None, None
        
        # Handle eval tournament
        if isinstance(self.state, EvalTournamentState):
            left_path, right_path = self.state.get_current_images()
            if left_path is None:
                return None, None
            left_img = Image.open(left_path)
            right_img = Image.open(right_path)
            return left_img, right_img
        
        # Handle regular tournament
        matchup = self.state.get_current_matchup()
        if matchup is None:
            return None, None
        
        left_idx, right_idx = matchup
        left_img = Image.open(self.state.image_paths[left_idx])
        right_img = Image.open(self.state.image_paths[right_idx])
        
        return left_img, right_img
    
    def _get_progress_text(self):
        """Get the progress text."""
        if self.state is None:
            return "⏳ Waiting for tournament to start..."
        if self.state.completed:
            if isinstance(self.state, EvalTournamentState):
                summary = self.state.get_results_summary()
                return f"✅ EVAL Complete! Model: {summary['model_wins']} wins ({summary['model_win_rate']:.0%}) | GPT-4.1: {summary['gpt_wins']} wins ({summary['gpt_win_rate']:.0%})"
            return f"✅ Tournament complete! All {len(self.state.matchups)} matchups judged."
        
        if isinstance(self.state, EvalTournamentState):
            return f"🔬 EVAL Round {self.state.round_num} — Matchup {self.state.current_matchup_idx + 1} of {len(self.state.matchups)} (Model vs GPT-4.1)"
        
        return f"🎮 Round {self.state.round_num} — Matchup {self.state.current_matchup_idx + 1} of {len(self.state.matchups)}"
    
    def _vote(self, choice: str):
        """Process a vote and return updated UI state."""
        if self.state is None:
            return None, None, "⏳ Waiting for tournament to start...", gr.update(visible=True), gr.update(visible=False)
        
        if self.state.completed:
            return None, None, self._get_progress_text(), gr.update(visible=False), gr.update(visible=True)
        
        # Record the vote
        self.state.record_vote(choice)
        
        if self.state.completed:
            return None, None, self._get_progress_text(), gr.update(visible=False), gr.update(visible=True)
        
        # Get next pair
        left_img, right_img = self._get_current_images()
        return left_img, right_img, self._get_progress_text(), gr.update(visible=True), gr.update(visible=False)
    
    def _vote_left(self):
        return self._vote('left')
    
    def _vote_right(self):
        return self._vote('right')
    
    def _refresh(self):
        """Refresh the current state (for when tournament starts)."""
        left_img, right_img = self._get_current_images()
        progress = self._get_progress_text()
        if self.state is None:
            return None, None, progress, gr.update(visible=True), gr.update(visible=False)
        if self.state.completed:
            return None, None, progress, gr.update(visible=False), gr.update(visible=True)
        return left_img, right_img, progress, gr.update(visible=True), gr.update(visible=False)
    
    def _build_ui(self):
        """Build the Gradio UI."""
        
        with gr.Blocks(title="Image Tournament") as demo:
            
            gr.Markdown("# 🎨 Image Tournament")
            
            progress = gr.Markdown(self._get_progress_text())
            
            with gr.Group(visible=True) as voting_group:
                with gr.Row():
                    with gr.Column(scale=1):
                        left_img = gr.Image(
                            label="⬅️ LEFT", 
                            type="pil"
                        )
                        left_btn = gr.Button("⬅️ Choose Left", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        right_img = gr.Image(
                            label="RIGHT ➡️", 
                            type="pil"
                        )
                        right_btn = gr.Button("Choose Right ➡️", variant="primary", size="lg")
                
                gr.Markdown("*Click a button to choose which image you prefer*")
                
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            
            complete_msg = gr.Markdown(
                "## 🎉 Tournament Complete!\n\nTraining will continue automatically...",
                visible=False
            )
            
            # Button click handlers
            left_btn.click(
                fn=self._vote_left,
                outputs=[left_img, right_img, progress, voting_group, complete_msg]
            )
            right_btn.click(
                fn=self._vote_right,
                outputs=[left_img, right_img, progress, voting_group, complete_msg]
            )
            refresh_btn.click(
                fn=self._refresh,
                outputs=[left_img, right_img, progress, voting_group, complete_msg]
            )
            
            # Load initial images
            demo.load(
                fn=self._refresh,
                outputs=[left_img, right_img, progress, voting_group, complete_msg]
            )
            
            # Auto-refresh every 2 seconds while waiting for tournament
            timer = gr.Timer(2)
            timer.tick(
                fn=self._refresh,
                outputs=[left_img, right_img, progress, voting_group, complete_msg]
            )
        
        return demo
    
    def start_tournament(
        self,
        round_num: int,
        image_paths: List[str],
        prompts: List[str]
    ) -> None:
        """Start a new regular tournament."""
        self.state = TournamentState(
            round_num=round_num,
            image_paths=image_paths,
            prompts=prompts
        )
    
    def start_eval_tournament(
        self,
        round_num: int,
        model_image_paths: List[str],
        model_prompts: List[str],
        gpt_image_paths: List[str],
        gpt_prompts: List[str]
    ) -> None:
        """Start a new eval tournament (model vs GPT-4.1)."""
        self.state = EvalTournamentState(
            round_num=round_num,
            model_image_paths=model_image_paths,
            model_prompts=model_prompts,
            gpt_image_paths=gpt_image_paths,
            gpt_prompts=gpt_prompts
        )
    
    def wait_for_completion(self, poll_interval: float = 0.5):
        """Block until tournament is complete, return win rates or eval results."""
        while self.state is None or not self.state.completed:
            time.sleep(poll_interval)
        
        if isinstance(self.state, EvalTournamentState):
            return self.state.get_results_summary()
        return self.state.get_win_rates()
    
    def start_server_background(self):
        """Start the Gradio server in a background thread."""
        
        def run_server():
            self.demo = self._build_ui()
            # Launch with share=True for public URL
            self.demo.launch(
                share=self.share,
                server_name="0.0.0.0",
                server_port=7860,
                quiet=False,  # Show the URL
                show_error=True,
            )
        
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        
        # Wait for server to start
        time.sleep(8)
        
        print("\n" + "="*60)
        print("🌐 TOURNAMENT SERVER STARTED")
        print("="*60)
        print("Look above for the public URL (https://xxxxx.gradio.live)")
        print("="*60 + "\n")


def run_tournament(
    round_num: int,
    image_paths: List[str],
    prompts: List[str],
    server: TournamentServer
) -> List[float]:
    """
    Run a complete tournament and return win rates.
    
    Args:
        round_num: Training round number
        image_paths: List of paths to generated images
        prompts: List of prompts that generated each image
        server: The tournament server instance
    
    Returns:
        List of win rates for each image (0.0 to 1.0)
    """
    
    n_matchups = len(list(combinations(range(len(image_paths)), 2)))
    
    print(f"\n{'='*60}")
    print(f"🎮 TOURNAMENT ROUND {round_num}")
    print(f"{'='*60}")
    print(f"   {len(image_paths)} images, {n_matchups} matchups")
    print(f"   Waiting for human feedback via web UI...")
    print(f"{'='*60}\n")
    
    # Start tournament
    server.start_tournament(round_num, image_paths, prompts)
    
    # Wait for completion
    win_rates = server.wait_for_completion()
    
    print(f"\n✅ Tournament {round_num} complete!")
    print("   Win rates:", [f"{wr:.2f}" for wr in win_rates])
    
    return win_rates


def run_eval_tournament(
    round_num: int,
    model_image_paths: List[str],
    model_prompts: List[str],
    gpt_image_paths: List[str],
    gpt_prompts: List[str],
    server: TournamentServer
) -> Dict:
    """
    Run an evaluation tournament (model vs GPT-4.1).
    
    Args:
        round_num: Training round number
        model_image_paths: Paths to model-generated images
        model_prompts: Prompts from trained model
        gpt_image_paths: Paths to GPT-4.1-generated images
        gpt_prompts: Prompts from GPT-4.1
        server: The tournament server instance
    
    Returns:
        Dict with evaluation results
    """
    
    n_matchups = len(model_image_paths) * len(gpt_image_paths)
    
    print(f"\n{'='*60}")
    print(f"🔬 EVALUATION ROUND {round_num} — Model vs GPT-4.1")
    print(f"{'='*60}")
    print(f"   {len(model_image_paths)} model images vs {len(gpt_image_paths)} GPT images")
    print(f"   {n_matchups} matchups (randomized left/right)")
    print(f"   Waiting for human feedback via web UI...")
    print(f"{'='*60}\n")
    
    # Start eval tournament
    server.start_eval_tournament(
        round_num, 
        model_image_paths, model_prompts,
        gpt_image_paths, gpt_prompts
    )
    
    # Wait for completion
    results = server.wait_for_completion()
    
    print(f"\n✅ Evaluation {round_num} complete!")
    print(f"   Model wins: {results['model_wins']} ({results['model_win_rate']:.0%})")
    print(f"   GPT-4.1 wins: {results['gpt_wins']} ({results['gpt_win_rate']:.0%})")
    
    return results


if __name__ == "__main__":
    # Test the server with dummy images
    import tempfile
    
    # Create dummy images
    temp_dir = tempfile.mkdtemp()
    image_paths = []
    prompts = []
    
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan']
    
    for i in range(8):
        img = Image.new('RGB', (256, 256), colors[i])
        path = os.path.join(temp_dir, f"test_{i}.png")
        img.save(path)
        image_paths.append(path)
        prompts.append(f"Test prompt {i}")
    
    # Run server
    server = TournamentServer(share=True)
    server.start_server_background()
    
    print("Starting test tournament...")
    win_rates = run_tournament(0, image_paths, prompts, server)
    print("Final win rates:", win_rates)
