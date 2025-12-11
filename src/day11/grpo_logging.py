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
    is_training: bool = True
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
    """
    
    # Create subdirectory
    subdir = "train" if is_training else "test"
    step_dir = os.path.join(output_dir, subdir, f"step_{step_num}")
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
    is_training: bool = True
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
    """
    
    # Create subdirectory
    subdir = "train" if is_training else "test"
    step_dir = os.path.join(output_dir, subdir, f"step_{step_num}")
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
    is_training: bool = True
):
    """
    Update or create metrics JSON file with average cosine similarity per step.
    
    Args:
        step_num: Step number
        avg_cosine_similarity: Average cosine similarity for this step
        output_dir: Base output directory
        is_training: True for training, False for eval
    """
    
    subdir = "train" if is_training else "test"
    subdir_path = os.path.join(output_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)
    
    metrics_file = os.path.join(subdir_path, "metrics.json")
    
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

