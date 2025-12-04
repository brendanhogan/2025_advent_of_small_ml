"""
Debug script to compare local vs vLLM generation outputs side-by-side.
"""
import torch
import argparse
import json
from main import parse_args, generate_local, generate_vllm
from math_dataset import load_math_dataset, format_math_problem, extract_math_answer
import llms
import utils
import vllm_client as v_c


def main():
    # Setup
    args = parse_args()
    args.num_chains = 1  # Generate just 1 completion per method for direct comparison
    args.eval_size = 10  # Test on 10 problems
    args.seed = 42  # Fixed seed
    
    utils.seed_everything(args.seed)
    
    # Load model and tokenizer
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, use_liger_model=False)
    
    # Load dataset
    train_ds, eval_ds = load_math_dataset(
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed
    )
    
    # Setup vLLM client if needed
    vllm_client = None
    if args.use_vllm:
        base_url = f"http://{args.vllm_host}:{args.vllm_port}"
        vllm_client = v_c.VLLMClient(base_url=base_url)
        vllm_client.init_communicator(device=model.device)
        print(f"Connected to vLLM server at {args.vllm_host}:{args.vllm_port}")
    
    # System prompt
    system_prompt = (
        "Think first and reason step by step. Put your reasoning within <think></think> tags. "
        "Then put your final answer within <answer></answer> tags. "
        "You must use both tags in this exact order: first <think>your reasoning</think>, then <answer>your answer</answer>."
        f"Note: Your reasoning may be cut off if it gets too long, but answer as best as you can if that happens."
    )
    
    # Compare on a few examples
    comparison_results = []
    
    for i, entry in enumerate(eval_ds):
        if i >= args.eval_size:
            break
            
        question = format_math_problem(entry)
        target_answer = extract_math_answer(entry)
        
        prompt_text, prompt_ids, prompt_mask = utils.format_prompt(system_prompt, question, tokenizer)
        
        print(f"\n{'='*100}")
        print(f"Problem {i+1}: {entry['subject']} Level {entry['level']}")
        print(f"Question: {question[:100]}...")
        print(f"Target Answer: {target_answer}")
        print(f"{'='*100}")
        
        # Generate with LOCAL
        print("\n[LOCAL GENERATION]")
        with torch.no_grad():
            _, local_prompt_ids, local_completion_ids, _, _, local_completions = generate_local(
                model, tokenizer, prompt_ids, prompt_mask, args
            )
        local_output = local_completions[0]
        local_extracted = utils.extract_answer(local_output)
        local_correct = eval_ds.score_answer(answer=local_extracted, entry=entry) if local_extracted else 0.0
        
        print(f"Output: {local_output[:200]}...")
        print(f"Extracted Answer: {local_extracted}")
        print(f"Correct: {local_correct == 1.0}")
        
        # Generate with vLLM if enabled
        if vllm_client:
            print("\n[vLLM GENERATION]")
            with torch.no_grad():
                _, vllm_prompt_ids, vllm_completion_ids, _, _, vllm_completions = generate_vllm(
                    vllm_client, prompt_text, tokenizer, args, model.device
                )
            vllm_output = vllm_completions[0]
            vllm_extracted = utils.extract_answer(vllm_output)
            vllm_correct = eval_ds.score_answer(answer=vllm_extracted, entry=entry) if vllm_extracted else 0.0
            
            print(f"Output: {vllm_output[:200]}...")
            print(f"Extracted Answer: {vllm_extracted}")
            print(f"Correct: {vllm_correct == 1.0}")
            
            # Compare tokenization
            print("\n[TOKENIZATION COMPARISON]")
            print(f"Prompt IDs match: {torch.equal(local_prompt_ids, vllm_prompt_ids)}")
            if not torch.equal(local_prompt_ids, vllm_prompt_ids):
                print(f"  Local prompt length: {local_prompt_ids.shape}")
                print(f"  vLLM prompt length: {vllm_prompt_ids.shape}")
                print(f"  First 10 local: {local_prompt_ids[0, :10].tolist()}")
                print(f"  First 10 vLLM: {vllm_prompt_ids[0, :10].tolist()}")
            
            print(f"\nCompletion IDs match: {torch.equal(local_completion_ids, vllm_completion_ids)}")
            if not torch.equal(local_completion_ids, vllm_completion_ids):
                print(f"  Local completion length: {local_completion_ids.shape}")
                print(f"  vLLM completion length: {vllm_completion_ids.shape}")
                print(f"  First 20 local: {local_completion_ids[0, :20].tolist()}")
                print(f"  First 20 vLLM: {vllm_completion_ids[0, :20].tolist()}")
            
            # Check if outputs are identical
            outputs_match = local_output.strip() == vllm_output.strip()
            print(f"\nOutputs match: {outputs_match}")
            
            comparison_results.append({
                "problem": i+1,
                "question": question,
                "target_answer": target_answer,
                "local": {
                    "output": local_output,
                    "extracted": local_extracted,
                    "correct": local_correct == 1.0,
                },
                "vllm": {
                    "output": vllm_output,
                    "extracted": vllm_extracted,
                    "correct": vllm_correct == 1.0,
                },
                "outputs_match": outputs_match,
            })
    
    # Save results
    if vllm_client:
        with open("debug_comparison.json", "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\n\nResults saved to debug_comparison.json")
        
        # Summary
        local_correct_count = sum(1 for r in comparison_results if r["local"]["correct"])
        vllm_correct_count = sum(1 for r in comparison_results if r["vllm"]["correct"])
        outputs_match_count = sum(1 for r in comparison_results if r["outputs_match"])
        
        print(f"\n{'='*100}")
        print(f"SUMMARY")
        print(f"{'='*100}")
        print(f"Local correct: {local_correct_count}/{len(comparison_results)}")
        print(f"vLLM correct:  {vllm_correct_count}/{len(comparison_results)}")
        print(f"Outputs identical: {outputs_match_count}/{len(comparison_results)}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()

