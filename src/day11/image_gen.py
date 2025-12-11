"""
Image generation using Replicate's Flux model
"""

import io
import os
import replicate
from PIL import Image
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def generate_image(prompt: str, output_path: str = None) -> Image.Image:
    """
    Generate a single image from a text prompt using Flux.
    
    Args:
        prompt: Text prompt for image generation
        output_path: Optional path to save the image
    
    Returns:
        PIL Image object
    """
    
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
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
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Save if path provided
    if output_path:
        image.save(output_path)
    
    return image


def generate_images_batch(
    prompts: List[str], 
    output_dir: str = None,
    round_num: int = 0
) -> Tuple[List[Image.Image], List[str]]:
    """
    Generate images for multiple prompts in parallel.
    
    Args:
        prompts: List of text prompts
        output_dir: Directory to save images (optional)
        round_num: Round number for naming files
    
    Returns:
        tuple: (list of PIL Images, list of saved paths)
    """
    
    images = [None] * len(prompts)
    paths = [None] * len(prompts)
    
    def generate_one(idx: int, prompt: str):
        if output_dir:
            path = os.path.join(output_dir, f"round_{round_num}_image_{idx}.png")
        else:
            path = None
        
        image = generate_image(prompt, path)
        return idx, image, path
    
    # Generate in parallel (Replicate handles rate limiting)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(generate_one, idx, prompt)
            for idx, prompt in enumerate(prompts)
        ]
        
        for future in as_completed(futures):
            idx, image, path = future.result()
            images[idx] = image
            paths[idx] = path
    
    return images, paths

