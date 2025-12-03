"""
Simple evaluation harness for CharXiv
- Training iterator: just iterates through images
- Eval class: evaluates model responses with pass@1 metrics using GPT-4o grading
"""

import json
import os
from collections import defaultdict
from openai import OpenAI
import sys
from tqdm import tqdm

# Load constants from CharXiv
sys.path.append('data')
from constants import (
    DESCRIPTIVE_GRADING_PREFIX, DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL,
    REASONING_GRADING_PREFIX, REASONING_GRADING_INST
)


def get_rubric(qid):
    """Get the grading rubric for a descriptive question ID"""
    if qid in [1]:
        return DESCRIPTIVE_GRADING_ICL['title']
    if qid in [2, 3, 4, 5, 6, 7]:
        return DESCRIPTIVE_GRADING_ICL['ocr']
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        return DESCRIPTIVE_GRADING_ICL['quant']
    if qid in [11]:
        return DESCRIPTIVE_GRADING_ICL['bool']
    if qid in [13]:
        return DESCRIPTIVE_GRADING_ICL['enum']
    if qid in [16]:
        return DESCRIPTIVE_GRADING_ICL['trend']
    if qid in [18]:
        return DESCRIPTIVE_GRADING_ICL['layout']
    return None


def get_qid_from_question(question_text):
    """Find question ID from question text"""
    for qid, qtext in DESCRIPTIVE_GRADING_QMAP.items():
        if qtext.lower() in question_text.lower() or question_text.lower() in qtext.lower():
            return qid
    return None


def grade_descriptive_gpt(client, question, ground_truth, response):
    """Grade a descriptive question using GPT-4o exactly like CharXiv"""
    qid = get_qid_from_question(question)
    if qid is None:
        # Fallback: use simple exact match if we can't find qid
        if str(ground_truth).strip().lower() == str(response).strip().lower():
            return 1
        return 0
    
    rubric = get_rubric(qid)
    if rubric is None:
        # Fallback
        if str(ground_truth).strip().lower() == str(response).strip().lower():
            return 1
        return 0
    
    # Build prompt exactly like CharXiv
    question_text = DESCRIPTIVE_GRADING_QMAP[qid]
    json_keys = '["extract_answer", "score"]'
    prefix = DESCRIPTIVE_GRADING_PREFIX\
        .replace("<|NUM_TRIPLETS|>", "1")\
        .replace("<|OVERARCHING_QUESTION|>", question_text)\
        .replace("<|JSON_KEYS|>", json_keys)
    
    grading_query = prefix + rubric + f"T1:\nResponse 1: {response}\nGround Truth 1: {ground_truth}\n\n"
    
    # Call GPT-4o
    result = get_gpt_result(client, grading_query)
    if result and 'score' in result:
        return result['score']
    return 0


def grade_reasoning_gpt(client, question, ground_truth, response, inst_category=1):
    """Grade a reasoning question using GPT-4o exactly like CharXiv"""
    # Use instruction category 1 (text-in-chart) as default
    # You can pass inst_category if you know it
    grading_inst = REASONING_GRADING_INST.get(inst_category, REASONING_GRADING_INST[1])
    
    grading_query = REASONING_GRADING_PREFIX + grading_inst\
        .replace("<|question|>", question)\
        .replace("<|ground_truth|>", ground_truth)\
        .replace("<|response|>", response)
    
    # Call GPT-4o
    result = get_gpt_result(client, grading_query)
    if result and 'score' in result:
        return result['score']
    return 0


def get_gpt_result(client, prompt, max_retries=10):
    """Get result from GPT-4o with retries, exactly like CharXiv"""
    curr_retries = 0
    max_tokens = 256
    while curr_retries < max_retries:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4.1",
            response_format={"type": "json_object"},
            n=1,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            seed=42,
        ).choices[0].message.content
        content = json.loads(response)
        return content
    return {'extract_answer': 'Failed to parse response', 'score': 0}


class TrainingIterator:
    """Simple iterator for training images"""
    
    def __init__(self, train_dir='train_images'):
        self.train_dir = train_dir
        self.image_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    
    def __iter__(self):
        for image_file in self.image_files:
            image_path = os.path.join(self.train_dir, image_file)
            yield image_path
    
    def __len__(self):
        return len(self.image_files)


class CharXivEval:
    """Evaluation class for CharXiv test set"""
    
    def __init__(self, test_data_path='test_data/test_set.json', llm=None, openai_api_key=None, num_images=None):
        with open(test_data_path, 'r') as f:
            all_test_data = json.load(f)
        
        # Limit number of images if specified
        if num_images is not None:
            self.test_data = all_test_data[:num_images]
        else:
            self.test_data = all_test_data
        
        self.llm = llm
        self.num_completions = 5
        
        # Setup GPT-4o grader
        if openai_api_key:
            self.grader_client = OpenAI(api_key=openai_api_key)
        else:
            self.grader_client = None
    
    def evaluate(self):
        """Run evaluation and return metrics"""
        
        all_descriptive_scores = []
        all_reasoning_scores = []
        per_question_scores = defaultdict(list)
        
        # Calculate total number of questions
        total_questions = 0
        for item in self.test_data:
            total_questions += len(item['descriptive_questions']) + 1  # +1 for reasoning question
        
        # Create progress bar
        pbar = tqdm(total=total_questions, desc="Evaluating questions")
        
        for item in self.test_data:
            figure_id = item['figure_id']
            image_path = item['image_path']
            descriptive_questions = item['descriptive_questions']
            reasoning_question = item['reasoning_question']
            
            # Evaluate descriptive questions
            for q_idx, q_data in enumerate(descriptive_questions):
                question = q_data['question']
                ground_truth = q_data['answer']
                
                # Get 5 completions
                completions = []
                for _ in range(self.num_completions):
                    response = self.llm.answer_question(image_path, question)
                    completions.append(response)
                
                # Calculate pass@1: did any of the 5 completions get it right?
                pass_at_1 = 0
                for response in completions:
                    if self.grader_client:
                        score = grade_descriptive_gpt(self.grader_client, question, ground_truth, response)
                    else:
                        # Fallback to simple match if no grader
                        if str(ground_truth).strip().lower() == str(response).strip().lower():
                            score = 1
                        else:
                            score = 0
                    if score == 1:
                        pass_at_1 = 1
                        break
                
                all_descriptive_scores.append(pass_at_1)
                question_key = f"descriptive_{q_idx}_{question[:50]}"
                per_question_scores[question_key].append(pass_at_1)
                
                # Update progress bar
                pbar.update(1)
            
            # Evaluate reasoning question
            question = reasoning_question['question']
            ground_truth = reasoning_question['answer']
            
            # Get 5 completions
            completions = []
            for _ in range(self.num_completions):
                response = self.llm.answer_question(image_path, question)
                completions.append(response)
            
            # Calculate pass@1
            pass_at_1 = 0
            for response in completions:
                if self.grader_client:
                    score = grade_reasoning_gpt(self.grader_client, question, ground_truth, response)
                else:
                    # Fallback to simple match if no grader
                    if str(ground_truth).strip().lower() == str(response).strip().lower():
                        score = 1
                    else:
                        score = 0
                if score == 1:
                    pass_at_1 = 1
                    break
            
            all_reasoning_scores.append(pass_at_1)
            question_key = f"reasoning_{question[:50]}"
            per_question_scores[question_key].append(pass_at_1)
            
            # Update progress bar
            pbar.update(1)
        
        # Close progress bar
        pbar.close()
        
        # Calculate metrics
        avg_descriptive = sum(all_descriptive_scores) / len(all_descriptive_scores) if all_descriptive_scores else 0
        avg_reasoning = sum(all_reasoning_scores) / len(all_reasoning_scores) if all_reasoning_scores else 0
        
        per_question_metrics = {}
        for question_key, scores in per_question_scores.items():
            per_question_metrics[question_key] = sum(scores) / len(scores) if scores else 0
        
        results = {
            'average_descriptive_pass_at_1': avg_descriptive,
            'average_reasoning_pass_at_1': avg_reasoning,
            'per_question_metrics': per_question_metrics,
            'total_descriptive_questions': len(all_descriptive_scores),
            'total_reasoning_questions': len(all_reasoning_scores)
        }
        
        return results


if __name__ == "__main__":
    import base64
    
    # Simple GPT-4.1 wrapper
    class GPT41LLM:
        def __init__(self, api_key):
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4.1"
        
        def answer_question(self, image_path, question):
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Call GPT-4.1 vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=512
            )
            return response.choices[0].message.content
    
    # Get API key from environment
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    # Create LLM and evaluator (limit to 2 images for testing)
    llm = GPT41LLM(api_key)
    evaluator = CharXivEval(llm=llm, openai_api_key=api_key, num_images=2)
    
    # Run evaluation
    print("Running evaluation with GPT-4.1...")
    print(f"Evaluating {len(evaluator.test_data)} images")
    results = evaluator.evaluate()
    
    # Print results
    print("\n=== Results ===")
    print(f"Descriptive pass@1: {results['average_descriptive_pass_at_1']:.4f}")
    print(f"Reasoning pass@1: {results['average_reasoning_pass_at_1']:.4f}")
    print(f"Total descriptive questions: {results['total_descriptive_questions']}")
    print(f"Total reasoning questions: {results['total_reasoning_questions']}")
