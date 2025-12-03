"""
Dataset loader for CharXiv images
"""

import os
import random
from typing import Any


class CharXivImageLoader:
    """Simple loader for CharXiv training/test images"""
    
    def __init__(self, image_dir: str, random: bool = False):
        self.image_dir = image_dir
        self.random = random
        self.current_index = 0
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
        
        # Prompt for image description
        self.prompt = "Describe this image in detail. Be specific about visual elements, colors, shapes, text, and any other important features."
    
    @staticmethod
    def get_adversary_prompt():
        """Get the prompt for adversary to generate scientific plot descriptions"""
        return "Describe a plot or chart that might be shown in a scientific publication. Be specific about the type of plot, what data it shows, colors, labels, and any other important visual features. You can choose any scientific subject or style."
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if self.current_index >= len(self.image_files):
            raise StopIteration
        
        if self.random:
            idx = random.randint(0, len(self.image_files) - 1)
        else:
            idx = self.current_index
            self.current_index += 1
        
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        return image_path
    
    def reset(self):
        self.current_index = 0

