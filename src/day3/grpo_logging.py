"""
Logging utilities for GRPO training
Saves images, creates PDFs, and tracks metrics
"""

import os
import json
from xml.sax.saxutils import escape
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def save_step_images(
    step_num: int,
    input_image_path: str,
    generated_images: list,
    descriptions: list,
    cosine_similarities: list,
    output_dir: str,
    is_training: bool = True,
    model_type: str = "base"
):
    """
    Save all images and text files for a training/eval step.
    
    Args:
        step_num: Step number
        input_image_path: Path to original input image
        generated_images: List of PIL Images (generated)
        descriptions: List of description strings
        cosine_similarities: List of cosine similarity scores
        output_dir: Base output directory
        is_training: True for training, False for eval
        model_type: "base" or "adversary"
    """
    
    # Create subdirectory
    subdir = "train" if is_training else "test"
    model_prefix = f"{model_type}_" if model_type != "base" else ""
    step_dir = os.path.join(output_dir, subdir, f"{model_prefix}step_{step_num}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Copy input image
    input_image = Image.open(input_image_path)
    input_image.save(os.path.join(step_dir, f"step_{step_num}_input.png"))
    
    # Save each generated image and its prompt
    for i, (gen_image, description, cosine) in enumerate(zip(generated_images, descriptions, cosine_similarities)):
        # Save image
        gen_image.save(os.path.join(step_dir, f"step_{step_num}_completion_{i}.png"))
        
        # Save prompt text file
        prompt_file = os.path.join(step_dir, f"step_{step_num}_completion_{i}_prompt.txt")
        with open(prompt_file, 'w') as f:
            f.write(description)
            f.write(f"\n\nCosine Similarity: {cosine:.4f}")


def create_step_pdf(
    step_num: int,
    input_image_path: str,
    generated_images: list,
    descriptions: list,
    cosine_similarities: list,
    output_dir: str,
    is_training: bool = True,
    model_type: str = "base"
):
    """
    Create PDF for a training/eval step with input image and sorted table of outputs.
    
    Args:
        step_num: Step number
        input_image_path: Path to original input image
        generated_images: List of PIL Images (generated)
        descriptions: List of description strings
        cosine_similarities: List of cosine similarity scores
        output_dir: Base output directory
        is_training: True for training, False for eval
        model_type: "base" or "adversary"
    """
    
    # Create subdirectory
    subdir = "train" if is_training else "test"
    model_prefix = f"{model_type}_" if model_type != "base" else ""
    step_dir = os.path.join(output_dir, subdir, f"{model_prefix}step_{step_num}")
    pdf_path = os.path.join(step_dir, f"step_{step_num}_results.pdf")
    
    # Sort by cosine similarity (highest first)
    sorted_data = sorted(
        zip(generated_images, descriptions, cosine_similarities),
        key=lambda x: x[2],
        reverse=True
    )
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"Step {step_num} - {'Training' if is_training else 'Evaluation'}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Original input image
    story.append(Paragraph("Original Input Image", styles['Heading2']))
    input_img = PlatypusImage(input_image_path, width=4*inch, height=4*inch)
    story.append(input_img)
    story.append(Spacer(1, 0.3*inch))
    
    # Table header
    story.append(Paragraph("Generated Images (sorted by cosine similarity)", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Create table data
    table_data = [["Image", "Generated Prompt", "Cosine Similarity"]]
    
    for gen_image, description, cosine in sorted_data:
        # Save image temporarily for PDF
        temp_img_path = os.path.join(step_dir, f"temp_{len(table_data)-1}.png")
        gen_image.save(temp_img_path)
        
        # Add to table
        img_cell = PlatypusImage(temp_img_path, width=2*inch, height=2*inch)
        # Escape HTML/XML special characters to prevent parsing errors
        desc_text = description[:200] + "..." if len(description) > 200 else description
        desc_text_escaped = escape(desc_text)
        desc_cell = Paragraph(desc_text_escaped, styles['Normal'])
        cosine_cell = Paragraph(f"{cosine:.4f}", styles['Normal'])
        
        table_data.append([img_cell, desc_cell, cosine_cell])
    
    # Create table
    table = Table(table_data, colWidths=[2.5*inch, 3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temp images
    for i in range(len(sorted_data)):
        temp_path = os.path.join(step_dir, f"temp_{i}.png")
        if os.path.exists(temp_path):
            os.remove(temp_path)


def update_metrics_json(
    step_num: int,
    avg_cosine_similarity: float,
    output_dir: str,
    is_training: bool = True,
    model_type: str = "base"
):
    """
    Update or create metrics JSON file with average cosine similarity per step.
    
    Args:
        step_num: Step number
        avg_cosine_similarity: Average cosine similarity for this step
        output_dir: Base output directory
        is_training: True for training, False for eval
        model_type: "base" or "adversary"
    """
    
    subdir = "train" if is_training else "test"
    model_prefix = f"{model_type}_" if model_type != "base" else ""
    subdir_path = os.path.join(output_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)
    
    metrics_file = os.path.join(subdir_path, f"{model_prefix}metrics.json")
    
    # Load existing metrics if they exist
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # Update with new step
    metrics[str(step_num)] = {
        "avg_cosine_similarity": avg_cosine_similarity
    }
    
    # Save
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_adversary_step_images(
    step_num: int,
    adversary_prompts: list,
    adversary_generated_images: list,
    base_descriptions: list,
    reconstructed_images: list,
    cosine_similarities: list,
    output_dir: str,
    is_training: bool = True
):
    """
    Save all images and text files for an adversary training/eval step.
    
    Args:
        step_num: Step number
        adversary_prompts: List of adversary prompt strings
        adversary_generated_images: List of PIL Images (from adversary prompts)
        base_descriptions: List of base model description strings
        reconstructed_images: List of PIL Images (from base descriptions)
        cosine_similarities: List of cosine similarity scores
        output_dir: Base output directory
        is_training: True for training, False for eval
    """
    
    subdir = "train" if is_training else "test"
    step_dir = os.path.join(output_dir, subdir, f"adversary_step_{step_num}")
    os.makedirs(step_dir, exist_ok=True)
    
    # Save each adversary generation
    for i, (adv_prompt, adv_img, base_desc, recon_img, cosine) in enumerate(
        zip(adversary_prompts, adversary_generated_images, base_descriptions,
            reconstructed_images, cosine_similarities)
    ):
        # Save adversary-generated image
        adv_img.save(os.path.join(step_dir, f"step_{step_num}_adversary_image_{i}.png"))
        
        # Save reconstructed image
        recon_img.save(os.path.join(step_dir, f"step_{step_num}_reconstructed_image_{i}.png"))
        
        # Save text file with all info
        info_file = os.path.join(step_dir, f"step_{step_num}_generation_{i}_info.txt")
        with open(info_file, 'w') as f:
            f.write(f"Adversary Prompt:\n{adv_prompt}\n\n")
            f.write(f"Base Model Description:\n{base_desc}\n\n")
            f.write(f"Cosine Similarity: {cosine:.4f}\n")
            f.write(f"Reward (1 - cosine): {1.0 - cosine:.4f}\n")


def create_adversary_step_pdf(
    step_num: int,
    adversary_prompts: list,
    adversary_generated_images: list,
    base_descriptions: list,
    reconstructed_images: list,
    cosine_similarities: list,
    output_dir: str,
    is_training: bool = True
):
    """
    Create PDF for an adversary training/eval step.
    
    Args:
        step_num: Step number
        adversary_prompts: List of adversary prompt strings
        adversary_generated_images: List of PIL Images (from adversary prompts)
        base_descriptions: List of base model description strings
        reconstructed_images: List of PIL Images (from base descriptions)
        cosine_similarities: List of cosine similarity scores
        output_dir: Base output directory
        is_training: True for training, False for eval
    """
    
    subdir = "train" if is_training else "test"
    step_dir = os.path.join(output_dir, subdir, f"adversary_step_{step_num}")
    pdf_path = os.path.join(step_dir, f"step_{step_num}_results.pdf")
    
    # Sort by reward (1 - cosine similarity), highest first (worst reconstruction = best for adversary)
    sorted_data = sorted(
        zip(adversary_prompts, adversary_generated_images, base_descriptions,
            reconstructed_images, cosine_similarities),
        key=lambda x: 1.0 - x[4],  # Sort by reward (1 - cosine)
        reverse=True
    )
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title = Paragraph(f"Adversary Step {step_num} - {'Training' if is_training else 'Evaluation'}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Table header
    story.append(Paragraph("Adversary Generations (sorted by reward, highest first)", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))
    
    # Create table data
    table_data = [["Adv Prompt", "Adv Image", "Base Desc", "Recon Image", "Cosine Sim", "Reward"]]
    
    for adv_prompt, adv_img, base_desc, recon_img, cosine in sorted_data:
        # Save images temporarily for PDF
        temp_adv_path = os.path.join(step_dir, f"temp_adv_{len(table_data)-1}.png")
        temp_recon_path = os.path.join(step_dir, f"temp_recon_{len(table_data)-1}.png")
        adv_img.save(temp_adv_path)
        recon_img.save(temp_recon_path)
        
        # Add to table
        adv_img_cell = PlatypusImage(temp_adv_path, width=1.5*inch, height=1.5*inch)
        recon_img_cell = PlatypusImage(temp_recon_path, width=1.5*inch, height=1.5*inch)
        
        # Escape text
        adv_prompt_text = adv_prompt[:100] + "..." if len(adv_prompt) > 100 else adv_prompt
        base_desc_text = base_desc[:100] + "..." if len(base_desc) > 100 else base_desc
        adv_prompt_escaped = escape(adv_prompt_text)
        base_desc_escaped = escape(base_desc_text)
        
        adv_prompt_cell = Paragraph(adv_prompt_escaped, styles['Normal'])
        base_desc_cell = Paragraph(base_desc_escaped, styles['Normal'])
        cosine_cell = Paragraph(f"{cosine:.4f}", styles['Normal'])
        reward_cell = Paragraph(f"{1.0 - cosine:.4f}", styles['Normal'])
        
        table_data.append([adv_prompt_cell, adv_img_cell, base_desc_cell, recon_img_cell, cosine_cell, reward_cell])
    
    # Create table
    table = Table(table_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1*inch, 1*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temp images
    for i in range(len(sorted_data)):
        temp_adv_path = os.path.join(step_dir, f"temp_adv_{i}.png")
        temp_recon_path = os.path.join(step_dir, f"temp_recon_{i}.png")
        if os.path.exists(temp_adv_path):
            os.remove(temp_adv_path)
        if os.path.exists(temp_recon_path):
            os.remove(temp_recon_path)


def update_adversary_reward_json(
    step_num: int,
    avg_reward: float,
    output_dir: str,
    is_training: bool = True
):
    """
    Update or create JSON file with average adversary reward per step.
    
    Args:
        step_num: Step number
        avg_reward: Average reward for this step
        output_dir: Base output directory
        is_training: True for training, False for eval
    """
    
    subdir = "train" if is_training else "test"
    subdir_path = os.path.join(output_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)
    
    metrics_file = os.path.join(subdir_path, "adversary_rewards.json")
    
    # Load existing metrics if they exist
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = {}
    
    # Update with new step
    metrics[str(step_num)] = {
        "avg_reward": avg_reward
    }
    
    # Save
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

