"""
Evaluate checkpoints on CharXiv reasoning questions
Evaluates base model and all checkpoints, saves detailed results per checkpoint
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from math import comb
from transformers import GenerationConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

import grpo_llms
import data_eval


class QwenLLMWrapper:
    """Wrapper for Qwen model to match the LLM interface expected by CharXivEval"""
    
    def __init__(self, model, tokenizer, device, max_new_tokens=512, temperature=0.7):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
    
    def answer_question(self, image_path, question):
        """Generate answer to question about image"""
        
        # Resize image to save memory
        image = Image.open(image_path).convert('RGB')
        max_size = 224
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            image.save(temp_path)
            image_path = temp_path
        
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions about charts and graphs.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question}
                ],
            },
        ]
        
        text = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        image_inputs, video_inputs = process_vision_info(conversation)
        
        prompt_inputs = self.tokenizer(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device).to(self.model.dtype)
        
        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.tokenizer.pad_token_id,
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **prompt_inputs,
                generation_config=generation_config
            )
        
        # Extract completion
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = generated_ids[0, prompt_length:]
        
        # Decode
        response = self.tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return response.strip()


def load_checkpoint(checkpoint_path, model_name, device):
    """Load model from checkpoint"""
    model, tokenizer = grpo_llms.get_llm_tokenizer(model_name, device)
    
    # Load checkpoint info to get model name
    info_path = os.path.join(checkpoint_path, "checkpoint_info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            # Use model_name from checkpoint if available, otherwise use passed model_name
            if 'model_name' in info:
                model_name = info['model_name']
    
    # Load model state dict
    model_path = os.path.join(checkpoint_path, "model.pt")
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: model.pt not found in {checkpoint_path}, using base model")
    
    return model, tokenizer


def find_checkpoints(output_dir):
    """Find all checkpoints in output_dir/ckpts/"""
    ckpt_dir = os.path.join(output_dir, 'ckpts')
    if not os.path.exists(ckpt_dir):
        return []
    
    checkpoints = []
    for item in os.listdir(ckpt_dir):
        item_path = os.path.join(ckpt_dir, item)
        if os.path.isdir(item_path) and item.startswith('ckpt-'):
            checkpoints.append((item, item_path))
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x[0].split('-')[1]) if x[0].split('-')[1].isdigit() else 0)
    
    return checkpoints


def evaluate_checkpoint(model, tokenizer, device, test_data_subset, openai_api_key, checkpoint_name):
    """Evaluate a single checkpoint on reasoning questions"""
    
    # Create LLM wrapper
    llm = QwenLLMWrapper(model, tokenizer, device, max_new_tokens=512, temperature=0.7)
    
    # Create evaluator (we'll manually evaluate reasoning questions only)
    grader_client = None
    if openai_api_key:
        from openai import OpenAI
        grader_client = OpenAI(api_key=openai_api_key)
    
    results = {
        'checkpoint_name': checkpoint_name,
        'overall_pass_at_1': 0.0,
        'total_questions': 0,
        'per_chart_results': []
    }
    
    all_scores = []
    
    print(f"Evaluating {checkpoint_name} on {len(test_data_subset)} images...")
    
    for chart_data in tqdm(test_data_subset, desc=f"Evaluating {checkpoint_name}"):
        image_path = chart_data['image_path']
        reasoning_question = chart_data['reasoning_question']
        question = reasoning_question['question']
        ground_truth = reasoning_question['answer']
        
        # Generate 10 completions
        completions = []
        for _ in range(10):
            response = llm.answer_question(image_path, question)
            completions.append(response)
        
        # Grade each completion
        scores = []
        for response in completions:
            if grader_client:
                score = data_eval.grade_reasoning_gpt(grader_client, question, ground_truth, response)
            else:
                # Fallback to simple match
                if str(ground_truth).strip().lower() == str(response).strip().lower():
                    score = 1
                else:
                    score = 0
            scores.append(score)
        
        # Calculate probabilistic pass@1
        n = len(scores)  # total completions
        c = sum(scores)  # number of correct completions
        k = 1  # pass@1
        
        # Edge cases
        if c == 0:
            pass_at_1 = 0.0
        elif k > n:
            pass_at_1 = 1.0
        elif k == 0:
            pass_at_1 = 0.0
        else:
            # Formula: 1 - C(n - c, k) / C(n, k)
            pass_at_1 = 1 - comb(n - c, k) / comb(n, k)
        
        all_scores.append(pass_at_1)
        
        # Store per-chart results
        chart_result = {
            'figure_id': chart_data['figure_id'],
            'image_path': image_path,
            'question': question,
            'ground_truth': ground_truth,
            'completions': completions,
            'scores': scores,
            'pass_at_1': pass_at_1
        }
        results['per_chart_results'].append(chart_result)
    
    # Calculate overall pass@1
    results['overall_pass_at_1'] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    results['total_questions'] = len(all_scores)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on CharXiv reasoning questions")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory containing checkpoints")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Base model name")
    parser.add_argument("--num_images", type=int, default=10, help="Number of test images to evaluate on")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key for grading (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    openai_api_key = args.openai_api_key or os.environ.get('OPENAI_API_KEY')
    if not openai_api_key:
        print("Warning: No OpenAI API key provided. Will use simple string matching for grading.")
    
    # Load test data
    test_data_path = 'test_data/test_set.json'
    if not os.path.exists(test_data_path):
        print(f"Error: Test data not found at {test_data_path}")
        print("Please run build_ds.py first to prepare the test data.")
        return
    
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
    
    # Select fixed subset of images (first num_images)
    test_data_subset = test_data[:args.num_images]
    print(f"Using {len(test_data_subset)} fixed images for evaluation")
    
    # Create output directory
    ckpt_tests_dir = os.path.join(args.output_dir, 'ckpt_tests')
    os.makedirs(ckpt_tests_dir, exist_ok=True)
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.output_dir)
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Evaluate base model first
    print("\n" + "="*50)
    print("Evaluating base model (unfinetuned)")
    print("="*50)
    
    base_model, base_tokenizer = grpo_llms.get_llm_tokenizer(args.model_name, args.device)
    base_results = evaluate_checkpoint(
        base_model, base_tokenizer, args.device, test_data_subset,
        openai_api_key, "base_model"
    )
    
    # Save base model results
    base_output_path = os.path.join(ckpt_tests_dir, "base_model.json")
    with open(base_output_path, 'w') as f:
        json.dump(base_results, f, indent=2)
    print(f"Saved base model results to {base_output_path}")
    print(f"Base model pass@1: {base_results['overall_pass_at_1']:.4f}")
    
    # Evaluate each checkpoint
    for ckpt_name, ckpt_path in checkpoints:
        print("\n" + "="*50)
        print(f"Evaluating checkpoint: {ckpt_name}")
        print("="*50)
        
        try:
            model, tokenizer = load_checkpoint(ckpt_path, args.model_name, args.device)
            ckpt_results = evaluate_checkpoint(
                model, tokenizer, args.device, test_data_subset,
                openai_api_key, ckpt_name
            )
            
            # Save checkpoint results
            ckpt_output_path = os.path.join(ckpt_tests_dir, f"{ckpt_name}.json")
            with open(ckpt_output_path, 'w') as f:
                json.dump(ckpt_results, f, indent=2)
            print(f"Saved checkpoint results to {ckpt_output_path}")
            print(f"{ckpt_name} pass@1: {ckpt_results['overall_pass_at_1']:.4f}")
            
            # Clean up model to save memory
            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {ckpt_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*50)
    print("Evaluation complete!")
    print(f"Results saved in {ckpt_tests_dir}")
    print("="*50)


if __name__ == "__main__":
    main()

