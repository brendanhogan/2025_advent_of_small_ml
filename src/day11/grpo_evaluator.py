"""
Reward evaluator for image description task
Reward = cosine similarity between original image and regenerated image from description
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import replicate
import io
import os
import tempfile


class RewardEvaluator(ABC):
    """Abstract base class for reward computation"""
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        image_path: str,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rewards for completions"""
        pass
    
    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores to labeled dictionary"""
        pass


class ImageDescriptionEvaluator(RewardEvaluator):
    """
    Evaluator for image description task.
    Reward = cosine similarity between original and regenerated image.
    """
    
    def __init__(self, dino_model_name='facebook/dinov2-large', device='cuda'):
        self.num_reward_functions = 1
        self.device = device
        
        # Load DINOv2 model for image embeddings
        self.dino_processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.dino_model = AutoModel.from_pretrained(dino_model_name)
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()
    
    def _generate_image_from_description(self, description: str, device: str) -> tuple[torch.Tensor, Image.Image]:
        """
        Generate image from description using Replicate flux-schnell, then get DINOv2 embedding.
        Returns: (embedding, PIL Image)
        """
        # Generate image using Replicate
        output = replicate.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": description,
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80,
                "num_inference_steps": 4
            }
        )
        
        # Get image data
        image_data = output[0].read()
        
        # Load image from bytes and resize to save memory
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        # Resize to max 224x224 to save memory (maintains aspect ratio, matches original code)
        max_size = 224
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Get DINOv2 embedding of generated image
        inputs = self.dino_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        
        return embedding, image
    
    def _get_image_embedding(self, image_path: str, device: str) -> torch.Tensor:
        """
        Get embedding of original image using DINOv2.
        """
        # Load image and resize to save memory
        image = Image.open(image_path).convert('RGB')
        # Resize to max 224x224 to save memory (maintains aspect ratio, matches original code)
        max_size = 224
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Process image
        inputs = self.dino_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embedding
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
            last_hidden_states = outputs.last_hidden_state

            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        return embedding
    
    def _cosine_similarity_reward(self, completions: List[List[Dict[str, str]]], image_path: str, device: str) -> tuple[List[float], List[Image.Image], List[str]]:
        """Compute cosine similarity reward for each completion. Returns (rewards, generated_images, descriptions)"""
        rewards = []
        generated_images = []
        descriptions = []
        
        # Get original image embedding
        original_embedding = self._get_image_embedding(image_path, device)
        original_embedding = F.normalize(original_embedding, p=2, dim=0)
        
        for completion in completions:
            description = completion[0]['content']
            descriptions.append(description)
            
            # Generate image from description
            generated_embedding, generated_image = self._generate_image_from_description(description, device)
            generated_images.append(generated_image)
            generated_embedding = F.normalize(generated_embedding, p=2, dim=0)
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                original_embedding.unsqueeze(0),
                generated_embedding.unsqueeze(0)
            ).item()
            
            # Reward is the similarity (range: -1 to 1, we want 0 to 1)
            reward = (similarity + 1.0) / 2.0  # Normalize to [0, 1]
            rewards.append(reward)
        
        return rewards, generated_images, descriptions
    
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        image_path: str,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float], List[Image.Image], List[str], List[float]]:
        """
        Compute rewards for completions.
        Returns: (rewards_per_func, metrics, generated_images, descriptions, cosine_similarities)
        """
        
        num_completions = len(completions)
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        
        # Compute cosine similarity rewards (returns images and descriptions too)
        similarity_scores, generated_images, descriptions = self._cosine_similarity_reward(completions, image_path, device)
        rewards_per_func[:, 0] = torch.tensor(similarity_scores, dtype=torch.float32, device=device)
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(0)
        
        # Get raw cosine similarities (before normalization)
        cosine_similarities = [(score * 2.0) - 1.0 for score in similarity_scores]  # Convert back from [0,1] to [-1,1]
        
        metrics = {
            "rewards/cosine_similarity": reward_per_func[0].item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
        }
        
        return rewards_per_func, metrics, generated_images, descriptions, cosine_similarities
    
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """Convert reward scores to labeled dictionary"""
        return {
            'cosine_similarity': reward_scores[0].item()
        }

