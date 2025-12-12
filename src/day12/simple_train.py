"""
Simple GRPO training with emotion-based rewards.

1. Model generates prompts for image generation
2. Flux generates images from those prompts
3. You view images in slideshow, webcam captures your reactions
4. Hume AI analyzes your facial expressions (all 48 emotions!)
5. Emotion deviation = reward signal for GRPO
6. Model learns to write prompts that make you react!
"""

import os
import io
import time
import json
import base64
import tempfile
import threading
import argparse
import random
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np
import requests

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Gradio temp directory
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/.cache/gradio")
os.makedirs(os.path.expanduser("~/.cache/gradio"), exist_ok=True)

import gradio as gr
import replicate


# ============================================================================
# CONFIG
# ============================================================================

META_PROMPTS = {
    "happy": "Write a short, creative prompt for generating a joyful image. Be original and unexpected.",
    "scary": "Write a short, creative prompt for generating a creepy or unsettling image. Be original.",
    "funny": "Write a short, creative prompt for generating a hilarious image. Find humor in absurd situations.",
}

SYSTEM_PROMPT = """You are a creative artist who writes prompts for image generation. 
Be original and vivid. Output ONLY the prompt, nothing else."""

# Key emotions to track (Hume returns ~48, these are most relevant)
KEY_EMOTIONS = [
    "Joy", "Amusement", "Surprise (positive)", "Interest", "Excitement",
    "Confusion", "Concentration", "Doubt", "Contemplation",
    "Fear", "Horror", "Anxiety", "Disgust",
    "Sadness", "Disappointment", "Boredom",
    "Anger", "Contempt", "Annoyance",
    "Calmness", "Satisfaction", "Admiration", "Awe"
]


# ============================================================================
# HUME AI EMOTION ANALYSIS (Streaming API - Real-time)
# ============================================================================

import asyncio
from hume import AsyncHumeClient
from hume.expression_measurement.stream.stream.types import Config, StreamFace


async def analyze_face_hume_async(image_path: str, api_key: str) -> Optional[Dict[str, float]]:
    """
    Send face image to Hume streaming API - real-time results (~1 second).
    """
    try:
        start = time.time()
        client = AsyncHumeClient(api_key=api_key)
        
        # Config specifying we want face emotion detection
        config = Config(face=StreamFace())
        
        async with client.expression_measurement.stream.connect() as socket:
            result = await socket.send_file(image_path, config=config)
        
        elapsed = time.time() - start
        
        # Parse result - extract emotions from face predictions
        if result and hasattr(result, 'face') and result.face and result.face.predictions:
            pred = result.face.predictions[0]
            if hasattr(pred, 'emotions') and pred.emotions:
                emotions = {e.name: e.score for e in pred.emotions}
                print(f"    📊 Hume: {len(emotions)} emotions in {elapsed:.1f}s")
                return emotions
        
        # Check for error message
        if hasattr(result, 'error') and result.error:
            print(f"    ❌ Hume error: {result.error[:100]}")
        else:
            print(f"    ⚠️ No face detected")
        return None
        
    except Exception as e:
        print(f"    ❌ Hume error: {e}")
        return None


def analyze_face_hume(image_path: str, api_key: str, timeout: int = 60) -> Optional[Dict[str, float]]:
    """
    Synchronous wrapper for async Hume streaming API.
    """
    return asyncio.run(analyze_face_hume_async(image_path, api_key))


def format_emotions(emotions: Dict[str, float], top_n: int = 8) -> str:
    """Format emotions for display - top N sorted by score."""
    if not emotions:
        return "No emotions detected"
    
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    lines = []
    for name, score in sorted_emotions:
        # Clean bar: 20 chars max, filled proportionally
        filled = int(score * 20)
        bar = "▓" * filled + "░" * (20 - filled)
        lines.append(f"{name:<20} {bar} {score:.2f}")
    return "```\n" + "\n".join(lines) + "\n```"


def format_emotions_compact(emotions: Dict[str, float], top_n: int = 5) -> str:
    """Compact emotion format for logs."""
    if not emotions:
        return "no face"
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ", ".join([f"{n}:{s:.2f}" for n, s in sorted_emotions])


def compute_emotion_deviation(baseline: Dict[str, float], current: Dict[str, float]) -> float:
    """
    Compute total emotional deviation from baseline.
    Higher = more emotional reaction to the image.
    """
    if not baseline or not current:
        return 0.0
    
    total_deviation = 0.0
    count = 0
    
    for emotion in KEY_EMOTIONS:
        if emotion in baseline and emotion in current:
            deviation = abs(current[emotion] - baseline[emotion])
            total_deviation += deviation
            count += 1
    
    # Also compute for any emotion that changed significantly
    for emotion, current_score in current.items():
        if emotion not in KEY_EMOTIONS and emotion in baseline:
            deviation = abs(current_score - baseline[emotion])
            if deviation > 0.1:  # Only count big changes
                total_deviation += deviation
                count += 1
    
    return total_deviation / max(count, 1)


# ============================================================================
# IMAGE GENERATION
# ============================================================================

def generate_image(prompt: str, output_path: str = None) -> Image.Image:
    """Generate image from prompt using Flux."""
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
            "go_fast": True,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "num_inference_steps": 4
        }
    )
    
    image_data = output[0].read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    if output_path:
        image.save(output_path)
    
    return image


# ============================================================================
# MODEL & GENERATION
# ============================================================================

def load_model(model_name: str, device: str):
    """Load model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    model.config.use_cache = False
    
    return model, tokenizer


def generate_prompts(
    model, tokenizer, meta_prompt: str, 
    num_prompts: int, temperature: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Generate multiple prompts from the model."""
    
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": meta_prompt}
    ]
    
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    
    # Batch for multiple generations
    batched = {k: v.repeat(num_prompts, *([1] * (v.dim() - 1))) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].size(1)
    
    with torch.no_grad():
        outputs = model.generate(
            **batched,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    prompt_ids = outputs[:, :prompt_length]
    completion_ids = outputs[:, prompt_length:]
    
    # Create completion mask
    is_eos = completion_ids == tokenizer.eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    seq_idx = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (seq_idx <= eos_idx.unsqueeze(1)).int()
    
    attention_mask = torch.cat([batched["attention_mask"], completion_mask], dim=1)
    
    prompts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    
    return outputs, prompt_ids, completion_ids, attention_mask, prompts


# ============================================================================
# GRPO LOSS
# ============================================================================

def get_per_token_logps(model, input_ids, attention_mask, num_completion_tokens):
    """Get log probabilities for completion tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Shift for next-token prediction
    
    target_ids = input_ids[:, -num_completion_tokens:]
    target_logits = logits[:, -num_completion_tokens:]
    
    log_probs = F.log_softmax(target_logits, dim=-1)
    token_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    return token_logps


def compute_grpo_loss(
    model, prompt_completion_ids, prompt_ids, completion_ids, 
    attention_mask, advantages
):
    """Compute GRPO loss."""
    completion_mask = attention_mask[:, prompt_ids.size(1):]
    num_completion_tokens = completion_ids.size(1)
    
    per_token_logps = get_per_token_logps(
        model, prompt_completion_ids, attention_mask, num_completion_tokens
    )
    
    # GRPO: weight log probs by advantages
    per_token_loss = -per_token_logps * advantages.unsqueeze(1)
    loss = ((per_token_loss * completion_mask).sum(1) / completion_mask.sum(1)).mean()
    
    return loss


# ============================================================================
# TRAINING STATE
# ============================================================================

@dataclass
class TrainingState:
    """Tracks training session state."""
    round_num: int
    image_paths: List[str]
    prompts: List[str]
    display_time: float = 2.0
    
    phase: str = "waiting"
    current_idx: int = 0
    
    # Full emotion data
    baseline_emotions: Dict[str, float] = field(default_factory=dict)
    image_emotions: List[Dict[str, float]] = field(default_factory=list)
    emotion_scores: List[float] = field(default_factory=list)
    
    completed: bool = False


# ============================================================================
# TRAINER
# ============================================================================

class EmotionTrainer:
    """Combined trainer + viewer."""
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hume_api_key = os.environ.get("HUME_API_KEY")
        
        self.state: Optional[TrainingState] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._waiting_for_user = threading.Event()
        self._frame_count = 0
        self._last_frame_time = 0
        
        # Load model
        print(f"Loading {args.model_name}...")
        self.model, self.tokenizer = load_model(args.model_name, self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.1
        )
        
        # Output directory
        os.makedirs(args.output_dir, exist_ok=True)
        self.log_path = os.path.join(args.output_dir, "training_log.json")
        self.log_data = {"rounds": []}
    
    def _save_frame(self, frame: np.ndarray, save_path: str = None) -> str:
        """Save frame to file."""
        if save_path is None:
            temp_dir = tempfile.mkdtemp()
            save_path = os.path.join(temp_dir, "frame.jpg")
        
        Image.fromarray(frame).save(save_path, "JPEG", quality=95)
        return save_path
    
    def _update_frame(self, webcam_frame):
        """Store latest webcam frame from Gradio."""
        if webcam_frame is not None:
            with self._frame_lock:
                self._latest_frame = webcam_frame.copy() if isinstance(webcam_frame, np.ndarray) else np.array(webcam_frame)
                self._frame_count += 1
                self._last_frame_time = time.time()
        return webcam_frame
    
    def _get_display(self):
        """Get current display state."""
        if self.state is None:
            return None, "⏳ **Waiting for round...**", "Emotions will appear here..."
        
        emotions_text = ""
        
        if self.state.phase == "waiting":
            return None, f"🎯 **Round {self.state.round_num}**\n\nClick **Start** when ready!", ""
        
        if self.state.phase == "baseline":
            if self.state.baseline_emotions:
                emotions_text = "**📊 Baseline Emotions:**\n\n" + format_emotions(self.state.baseline_emotions, 8)
            else:
                emotions_text = "Capturing baseline..."
            return None, "📸 **Capturing baseline...**\n\nLook at camera neutrally", emotions_text
        
        if self.state.phase == "slideshow":
            if self.state.current_idx < len(self.state.image_paths):
                img = Image.open(self.state.image_paths[self.state.current_idx])
                prompt = self.state.prompts[self.state.current_idx]
                status = f"🎬 **Image {self.state.current_idx + 1}/{len(self.state.image_paths)}**\n\n_{prompt[:80]}_"
                
                # Show latest emotions and scores
                if self.state.image_emotions:
                    latest = self.state.image_emotions[-1]
                    emotions_text = f"**🎭 Current Emotions:**\n\n{format_emotions(latest, 8)}\n\n"
                    emotions_text += "**📈 Scores so far:**\n"
                    for i, score in enumerate(self.state.emotion_scores):
                        emotions_text += f"Image {i+1}: **{score:.4f}**\n"
                
                return img, status, emotions_text
        
        if self.state.phase == "complete":
            msg = f"✅ **Round {self.state.round_num} Complete!**\n\n"
            msg += "**Deviation Scores:**\n"
            for i, (score, prompt) in enumerate(zip(self.state.emotion_scores, self.state.prompts)):
                msg += f"\n{i+1}. **{score:.4f}** - _{prompt[:50]}..._"
            
            mean = sum(self.state.emotion_scores) / len(self.state.emotion_scores) if self.state.emotion_scores else 0
            msg += f"\n\n**Mean: {mean:.4f}**"
            
            # Show emotion summary
            if self.state.image_emotions:
                emotions_text = "**Final emotions detected:**\n\n"
                for i, emo in enumerate(self.state.image_emotions):
                    if emo:
                        emotions_text += f"**Image {i+1}:** {format_emotions_compact(emo)}\n"
            
            return None, msg, emotions_text
        
        return None, "...", ""
    
    def _run_slideshow(self):
        """Run slideshow and capture emotions - ONE capture per image."""
        if self.state is None:
            return
        
        round_dir = os.path.join(self.args.output_dir, f"round_{self.state.round_num:04d}")
        
        # Use neutral baseline
        self.state.baseline_emotions = {emotion: 0.1 for emotion in KEY_EMOTIONS}
        self.state.phase = "slideshow"
        
        for i in range(len(self.state.image_paths)):
            # === STEP 1: Show the image ===
            self.state.current_idx = i
            print(f"\n  🖼️ [{i+1}/{len(self.state.image_paths)}] {self.state.prompts[i][:60]}...")
            
            # === STEP 2: Wait for reaction (user views image) ===
            time.sleep(self.state.display_time)
            
            # === STEP 3: Capture ONE frame ===
            with self._frame_lock:
                frame = self._latest_frame.copy() if self._latest_frame is not None else None
            
            if frame is None:
                print(f"    ⚠️ No webcam frame")
                self.state.image_emotions.append({})
                self.state.emotion_scores.append(0.0)
                continue
            
            # Save the captured frame
            reaction_path = os.path.join(round_dir, f"reaction_{i}.jpg")
            self._save_frame(frame, reaction_path)
            print(f"    📸 Captured face, analyzing...")
            
            # === STEP 4: Analyze with Hume (blocking) ===
            emotions = analyze_face_hume(reaction_path, self.hume_api_key) or {}
            self.state.image_emotions.append(emotions)
            
            if emotions:
                score = compute_emotion_deviation(self.state.baseline_emotions, emotions)
                self.state.emotion_scores.append(score)
                print(f"    ✅ Score: {score:.4f} | {format_emotions_compact(emotions)}")
            else:
                self.state.emotion_scores.append(0.0)
                print(f"    ❌ No face detected")
            
            # Brief pause before next image
            time.sleep(0.5)
        
        self.state.phase = "complete"
        self.state.completed = True
        self._waiting_for_user.set()
        
        print(f"\n  ✅ Slideshow complete!")
    
    def _start_slideshow(self):
        """Start button handler."""
        if self.state and self.state.phase == "waiting":
            thread = threading.Thread(target=self._run_slideshow, daemon=True)
            thread.start()
        return self._get_display()
    
    def train_round(self, round_num: int):
        """Run one training round."""
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")
        
        meta_prompt = META_PROMPTS[self.args.mode]
        
        # 1. Generate prompts
        print("\n📝 Generating prompts...")
        prompt_completion_ids, prompt_ids, completion_ids, attention_mask, prompts = generate_prompts(
            self.model, self.tokenizer, meta_prompt,
            self.args.num_images, self.args.temperature, self.device
        )
        prompts = [p.strip() for p in prompts]
        
        for i, p in enumerate(prompts):
            print(f"  [{i}] {p[:70]}...")
        
        # 2. Generate images
        print(f"\n🎨 Generating {len(prompts)} images...")
        round_dir = os.path.join(self.args.output_dir, f"round_{round_num:04d}")
        os.makedirs(round_dir, exist_ok=True)
        
        image_paths = []
        for i, prompt in enumerate(prompts):
            path = os.path.join(round_dir, f"image_{i}.png")
            generate_image(prompt, path)
            image_paths.append(path)
            print(f"  ✓ Image {i+1}/{len(prompts)}")
        
        # 3. Setup state for slideshow
        self.state = TrainingState(
            round_num=round_num,
            image_paths=image_paths,
            prompts=prompts,
            display_time=self.args.display_time
        )
        self._waiting_for_user.clear()
        
        print("\n⏳ Click 'Start Round' in the UI...")
        
        # 4. Wait for slideshow to complete
        self._waiting_for_user.wait()
        
        # 5. Compute rewards and advantages
        rewards = torch.tensor(self.state.emotion_scores, dtype=torch.float32, device=self.device)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-4
        advantages = (rewards - mean_reward) / std_reward
        
        print(f"\n📊 Results:")
        print(f"  Rewards: {[f'{r:.3f}' for r in rewards.tolist()]}")
        print(f"  Mean: {mean_reward.item():.4f}, Std: {std_reward.item():.4f}")
        
        # 6. GRPO update
        loss = compute_grpo_loss(
            self.model, prompt_completion_ids, prompt_ids, 
            completion_ids, attention_mask, advantages
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        print(f"  Loss: {loss.item():.4f}")
        
        # 7. Save detailed log
        round_log = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "prompts": prompts,
            "emotion_scores": self.state.emotion_scores,
            "mean_reward": mean_reward.item(),
            "std_reward": std_reward.item(),
            "loss": loss.item(),
            "baseline_emotions": self.state.baseline_emotions,
            "image_emotions": self.state.image_emotions,
        }
        self.log_data["rounds"].append(round_log)
        
        with open(self.log_path, "w") as f:
            json.dump(self.log_data, f, indent=2)
        
        # Save round data
        with open(os.path.join(round_dir, "round_data.json"), "w") as f:
            json.dump(round_log, f, indent=2)
        
        print(f"\n💾 Saved to {round_dir}")
        
        return loss.item(), mean_reward.item()
    
    def run(self):
        """Main training loop with Gradio UI."""
        
        def training_loop():
            """Background training loop."""
            time.sleep(5)  # Wait for Gradio to start
            
            for round_num in range(self.args.num_rounds):
                self.train_round(round_num)
                torch.cuda.empty_cache()
            
            print("\n" + "="*60)
            print("🎉 TRAINING COMPLETE!")
            print(f"Logs saved to: {self.args.output_dir}")
            print("="*60)
        
        # Start training in background
        train_thread = threading.Thread(target=training_loop, daemon=True)
        train_thread.start()
        
        # Launch Gradio
        with gr.Blocks(title="Emotion GRPO Training") as demo:
            gr.Markdown("# 🎭 Emotion GRPO Training")
            gr.Markdown(f"**Mode:** {self.args.mode} | **Images per round:** {self.args.num_images} | **Display time:** {self.args.display_time}s")
            
            with gr.Row():
                with gr.Column(scale=2):
                    display_image = gr.Image(type="pil", height=450, show_label=False)
                    status = gr.Markdown("⏳ Loading model...")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 Webcam")
                    # Show the processed frame to create a feedback loop
                    webcam = gr.Image(sources=["webcam"], type="numpy", height=200, streaming=True, label="Input")
                    webcam_mirror = gr.Image(type="numpy", height=100, label="Captured", visible=True)
                    start_btn = gr.Button("▶️ Start Round", variant="primary", size="lg")
                    gr.Markdown("---")
                    gr.Markdown("### 🎭 Emotions")
                    emotions = gr.Markdown("Emotions will appear here...")
            
            # Stream webcam frames to backend
            webcam.stream(fn=self._update_frame, inputs=[webcam], outputs=[webcam_mirror])
            start_btn.click(fn=self._start_slideshow, outputs=[display_image, status, emotions])
            
            timer = gr.Timer(0.5)
            timer.tick(fn=self._get_display, outputs=[display_image, status, emotions])
        
        demo.launch(share=not self.args.no_share, server_name="0.0.0.0", server_port=7860)


def main():
    parser = argparse.ArgumentParser(description="GRPO training with emotion rewards")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="emotion_run_2")
    parser.add_argument("--mode", type=str, default="funny", choices=["happy", "scary", "funny"])
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--display_time", type=float, default=3.0)  # Longer for Hume processing
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--no_share", action="store_true")
    args = parser.parse_args()
    
    print("="*60)
    print("🎭 EMOTION GRPO TRAINING")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Mode: {args.mode}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Images per round: {args.num_images}")
    print(f"Display time: {args.display_time}s")
    print(f"Hume API: {'✓' if os.environ.get('HUME_API_KEY') else '✗ SET HUME_API_KEY!'}")
    print(f"Replicate: {'✓' if os.environ.get('REPLICATE_API_TOKEN') else '✗ SET REPLICATE_API_TOKEN!'}")
    print("="*60)
    
    trainer = EmotionTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
