# Day 6: Steering Toward Novelty - Can We Guide LLMs to Generate Breakthrough Ideas?

## The Thought Experiment

Here's something that's been bugging me: imagine there's a breakthrough scientific discovery waiting to be made - some novel idea that would fundamentally change how we think about AI, or physics, or whatever field. That discovery, once made, could be summarized in an abstract. A few hundred words that capture the essence of something revolutionary.

Now here's the interesting part: that abstract is just a sequence of tokens. It's a valid path through the language model's probability space - every token transition is something the model has seen before, every phrase is grammatically correct, every concept is one the model understands. The abstract exists, in some sense, on the manifold of possible texts the LLM knows about.

But here's the problem: even though that breakthrough abstract is a valid sequence, the LLM would almost certainly never generate it naturally. Why? Because LLMs sample in a way that reverts to the mean. They generate text that's coherent, reasonable, and... plain. They follow the well-trodden paths, the high-probability transitions, the safe choices. They don't venture into the weird corners of their own knowledge space where novel ideas might live.

So the question becomes: if these breakthrough ideas are valid paths through the model's knowledge, how do we guide the model to actually take those paths? How do we steer it away from the mean and toward novelty?

## The Approach: Steering Vectors for Creativity

This is where steering vectors come in. The idea is simple but powerful: what if we could learn a direction in activation space that points toward "more creative" or "more novel" text? 

The approach works like this:

1. **Collect examples of creative vs. plain text**: We use award-winning research paper abstracts (from ICML, ICLR, NeurIPS) as examples of highly creative, impactful writing. These are abstracts that won best paper awards - they represent work that the community recognized as truly novel and significant.

2. **Generate baseline abstracts**: For each award-winning abstract, we ask the model to generate its own abstract on the same topic (using a prompt like "Generate an abstract that would win best paper..."). These generated abstracts are typically more generic, less novel - they're what the model naturally produces.

3. **Extract activation differences**: We pass both the real award-winning abstract and the generated abstract through the model, collecting activations at each transformer layer. The difference between these activations - `real_activations - generated_activations` - gives us a "steering vector" that points in the direction of more creative writing.

4. **Average into a creativity vector**: We average steering vectors across many examples to get a general "creativity vector" that can be applied to any generation.

5. **Apply during generation**: When generating new text, we add this creativity vector to the hidden states at each layer, steering the model toward more novel outputs.

The key insight is that we're not teaching the model new facts - we're nudging it to explore different regions of its existing knowledge space. We're guiding it to take paths it knows about but wouldn't naturally choose.

## Setup

The repository includes a dataset of award-winning abstracts in `abstracts.json`. This file contains award-winning research paper abstracts from ICML, ICLR, and NeurIPS (best paper awards) that serve as examples of highly creative, impactful writing.

The file has this structure:

```json
{
  "abstracts": [
    {
      "title": "Paper Title",
      "abstract": "The abstract text..."
    },
    ...
  ]
}
```

You can also collect your own abstracts from conference proceedings and use the same format.

## Building the Creativity Vector

To build the steering vector from your abstracts:

```bash
uv run python build_creativity_vectors.py \
    --abstracts_path abstracts.json \
    --output_path creativity_vector.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --train_split 0.8 \
    --max_examples 50
```

This will:
- Load your abstracts and split into train/test (80/20)
- For each training abstract, generate a baseline abstract and compute the steering vector
- Average all steering vectors to create a general creativity vector
- Save the creativity vector and test set for evaluation

Key arguments:
- `--abstracts_path`: Path to your abstracts JSON file
- `--output_path`: Where to save the creativity vector (default: `creativity_vector.json`)
- `--model_name`: Model to use (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--train_split`: Fraction for training (default: 0.8)
- `--max_examples`: Limit training examples (useful for quick experiments)

## Evaluating the Creativity Vector

Once you have a creativity vector, you can evaluate it:

```bash
uv run python eval_creativity_vectors.py \
    --creativity_vector_path creativity_vector.json \
    --test_set_path creativity_vector_test_set.json \
    --steering_weight 0.05 \
    --openai_api_key $OPENAI_API_KEY
```

The evaluation results are saved to `creativity_eval_results.json`, which contains detailed results from all three evaluation parts, including the judge's comparisons and win rates.

This runs a three-part evaluation:

1. **Part 1**: Generate alternating base and steering abstracts (saves to JSON for manual inspection)
2. **Part 2**: Compare base vs. steering against real test abstracts (GPT-4o judges which is better)
3. **Part 3**: Direct comparison of base vs. steering (GPT-4o judges without context)

The evaluation results are saved to `creativity_eval_results.json`, which contains detailed results from all three parts, including the judge's comparisons, win rates, and all generated abstracts. You can inspect this file to see the quantitative results and read the actual generated abstracts.

The evaluation uses GPT-4o as a judge to determine which abstract demonstrates "more truly human creativity and impactful ideas." This is intentionally subjective - we're not measuring factual correctness, but rather novelty, insight, and the kind of creativity that leads to breakthroughs.

Key arguments:
- `--steering_weight`: How strongly to apply the steering vector (default: 0.05, try 0.01-0.2)
- `--num_pairs_part1`: Number of base/steering pairs for Part 1 (default: 5)
- `--num_rounds_part2`: Rounds per test abstract for Part 2 (default: 3)
- `--num_rounds_part3`: Rounds for Part 3 (default: 3)
- `--skip_part1`, `--skip_part2`, `--skip_part3`: Skip specific parts

## The Evaluation Challenge

Here's the honest truth: evaluating whether steering vectors actually produce "more creative" or "more novel" text is really hard. We tried using GPT-4o as a judge, asking it to compare base vs. steering abstracts and determine which demonstrates more "truly human creativity and impactful ideas." But even with a sophisticated judge, it's subjective.

The abstracts generated with steering do feel different when you read them - they tend to be more ambitious, use more novel framing, explore less conventional angles. But whether that translates to actual scientific breakthroughs? That's the million-dollar question.

What we can say is that this approach provides a mechanism for encouraging the model to explore different regions of its knowledge space. And that, in itself, is interesting. If we can guide models to write in specific ways - to be more novel, more creative, more exploratory - that could be incredibly useful for models trying to do scientific work.

## Potential Applications

The idea of steering toward novelty isn't just about generating better abstracts. It's about:

- **Scientific discovery**: Could we guide models to propose more novel hypotheses, explore unconventional research directions, or make unexpected connections?

- **Creative writing**: What if we could steer models toward more original narratives, less clichéd prose, more surprising plot developments?

- **Problem-solving**: Can we nudge models to consider solutions they wouldn't naturally think of - to explore the weird corners of their knowledge where novel approaches might live?

- **Research assistance**: Models that can generate more novel research directions, more creative experimental designs, or more insightful questions could be powerful tools for scientists.

The key insight is that we're not adding new knowledge to the model - we're helping it access knowledge it already has but wouldn't naturally use. We're giving it permission to take the weird paths, to explore the low-probability but high-value regions of its own understanding.

## Results

The results are very quantitative and encouraging! Using GPT-4o as a judge to evaluate which abstracts demonstrate "more truly human creativity and impactful ideas," the steering vectors show clear improvements:

**Part 2: Comparing base vs steering against real test abstracts**
- Base win rate: 32.50%
- Steering win rate: 67.50%

**Part 3: Direct comparison of base vs steering**
- Base win rate: 10.00%
- Steering win rate: 90.00%

The LLM judge consistently preferred the steering-generated abstracts as more novel and creative. This is a strong signal that the steering vectors are successfully guiding the model toward more exploratory, less conventional outputs.

Of course, this evaluation is subjective by design - we're not measuring factual correctness or even coherence (both base and steering abstracts are coherent). We're trying to measure something much fuzzier: novelty, insight, the kind of creativity that leads to breakthroughs. But the fact that a sophisticated judge consistently prefers the steering outputs suggests we're on the right track.

The steering vectors do change the output in measurable ways - abstracts generated with steering tend to be more ambitious, use more novel framing, and explore less conventional angles. Whether that translates to actual scientific breakthroughs? That's the open question. But we now have quantitative evidence that we can guide models to explore different regions of their knowledge space in ways that are perceived as more creative.

## The Bigger Picture

This experiment is really about a fundamental question: how do we get AI systems to be more creative? Not just to generate text that sounds creative, but to actually explore novel ideas, make unexpected connections, and venture into the weird corners of their knowledge where breakthroughs might live.

The thought experiment that started this - that breakthrough ideas exist as valid paths through the model's knowledge space, but paths the model would never naturally take - suggests that steering might be part of the answer. Not the whole answer, but a tool for nudging models toward the novel, the unexpected, the potentially revolutionary.

And if we can do that, even imperfectly, that opens up a lot of possibilities for AI-assisted scientific discovery, creative work, and problem-solving. The breakthrough might be out there, waiting in the model's knowledge space. We just need to figure out how to guide the model to find it.
