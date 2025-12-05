# Day 5: Bayesian Optimization for Steering Vector Weights

![Day 5](figs/day5.png)

## What's This About?

This is an extension of Day 4's work on steering vectors. Once you have steering vectors that capture the difference between a base model and a finetuned model, an interesting question emerges: **where should you apply them, and at what weight?**

In Day 4, we applied the same steering weight uniformly across all layers. But different layers might benefit from different amounts of steering. Early layers handle low-level feature extraction, while late layers handle high-level reasoning - maybe they need different steering strengths?

This is a classic optimization problem: we have a 4-dimensional search space (one weight per layer group), and each evaluation is expensive (running the model on an eval set). This is exactly where **Bayesian Optimization** shines.

Bayesian Optimization is a powerful technique for optimizing expensive black-box functions. It uses a **Gaussian Process** to model the objective function (in our case, pass@1 performance) and an acquisition function to intelligently select the next point to evaluate. Instead of random search or grid search, it learns from previous evaluations to focus on promising regions of the search space.

We thought it'd be fun to apply Bayesian Optimization here to:
1. Find the best weight assignment for each layer group
2. See what we can learn from the optimal weights - do early layers need more steering? Late layers? Are there interesting interactions?
3. Visualize how the optimization process explores the 4D space

The results are fascinating - the optimal weights tell us something about which parts of the model benefit most from steering, and the Gaussian Process gives us uncertainty estimates that help us understand how confident we are in different regions of the weight space.

## Setup

This project builds on Day 4, so you'll need:
- A trained model checkpoint (from Day 4)
- Steering vectors (built using `build_steering_vectors.py` from Day 4)
- The same dependencies: PyTorch, Transformers, etc.

Additionally, we use:
- `scikit-optimize` (skopt) for Bayesian optimization
- Matplotlib for visualizations
- Pillow for creating GIFs

## Running Bayesian Optimization

To find optimal steering weights using Bayesian optimization:

```bash
uv run python bayesian_optimize_steering.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --finetuned_checkpoint best_model \
  --steering_vectors_path steering_vectors.json \
  --output_dir bayesian_opt_results \
  --n_iterations 50 \
  --n_initial_points 5
```

This script:
1. Divides the 28 transformer layers into 4 groups (early, mid-early, mid-late, late)
2. Uses Bayesian optimization to search for optimal weights for each group
3. Evaluates each candidate configuration on the eval set
4. Creates visualizations showing the optimization progress
5. Saves the best configuration and full results

Key arguments:
- `--base_model_name`: Base model from HuggingFace
- `--finetuned_checkpoint`: Path to your trained checkpoint
- `--steering_vectors_path`: Path to steering vectors JSON file
- `--output_dir`: Where to save results and plots (default: `bayesian_opt_results`)
- `--n_iterations`: Number of Bayesian optimization iterations (default: 50)
- `--n_initial_points`: Number of random initial points before optimization starts (default: 5)
- `--eval-size`: Number of eval problems (default: 20)
- `--num_completions_eval`: Completions per problem (default: 20)
- `--resume_from`: Path to `optimization_results.json` to resume a previous run

The optimization can take a while since each evaluation requires running the model. The script saves progress after each iteration, so you can resume if interrupted.

## Visualizing Results

The optimization script automatically generates several types of visualizations:

### 1. 1D Gaussian Process Evolution

Shows how the GP models each layer group's weight over time:

```bash
uv run python plot_bayesian_optimization.py \
  --results-path bayesian_opt_results/final_results.json \
  --output-dir bayesian_opt_results/gp_plots
```

This creates plots for each iteration showing:
- The GP mean prediction (where it thinks performance is best)
- Uncertainty bands (how confident the GP is)
- Observed data points (actual evaluations)

### 2. 2D Pairwise Plots

Shows all pairwise combinations of the 4 dimensions:

```bash
uv run python plot_bayesian_optimization_2d.py \
  --results-path bayesian_opt_results/final_results.json \
  --output-dir bayesian_opt_results/gp_plots_2d
```

This creates a 2x3 grid showing:
- GP mean contours (predicted performance)
- Uncertainty visualization (darker = more confident)
- Scatter points colored by actual performance

Useful for understanding interactions between layer groups.

### 3. Parallel Coordinates Plot

Another way to visualize the 4D space:

```bash
uv run python plot_bayesian_optimization_parallel.py \
  --results-path bayesian_opt_results/final_results.json \
  --output-dir bayesian_opt_results/gp_plots_parallel
```

Shows all 4 dimensions on parallel axes, with lines connecting each configuration. High-performing configurations stand out visually.

### 4. Creating GIFs

To create animated GIFs of the optimization process:

```bash
uv run python create_gif.py \
  --frames-dir bayesian_opt_results/gp_plots \
  --pattern "gp_evolution_iter_*.png" \
  --output-path bayesian_opt_results/gp_optimization.gif
```

Do the same for 2D and parallel plots to create animated visualizations.

## Understanding the Results

After optimization completes, check `bayesian_opt_results/final_results.json` for:
- `best_weights`: The optimal weight for each layer group
- `best_score`: Performance with optimal weights
- `baseline_score`: Base model performance
- `finetuned_score`: Fully finetuned model performance
- `optimization_history`: Full history of all evaluations

The optimal weights tell us something interesting:
- **Which layers benefit most from steering?** If early layers have high optimal weights, maybe low-level feature extraction is where the finetuning helps most. If late layers have high weights, maybe high-level reasoning is the key.
- **Are there interactions?** The 2D plots show if certain layer groups work well together.
- **How close can we get to finetuned performance?** The best score tells us if steering vectors can match full finetuning.

## How It Works

### Layer Grouping

We divide the 28 transformer layers into 4 groups of 7 layers each:
- **Early** (layers 0-6): Low-level feature extraction
- **Mid-Early** (layers 7-13): Mid-level processing
- **Mid-Late** (layers 14-20): Higher-level reasoning
- **Late** (layers 21-27): Final output processing

Each group gets its own steering weight, allowing us to tune how much steering to apply at different depths.

### Gaussian Processes

A Gaussian Process is a non-parametric model that can represent smooth functions. Given some observations (weight configurations and their performance), the GP:
- Predicts performance at unseen configurations (the mean)
- Quantifies uncertainty (the variance)

The GP uses a kernel (we use a Matern kernel) to model how similar configurations should have similar performance. This allows the GP to generalize from observed points to the entire search space.

### Bayesian Optimization

Bayesian Optimization uses the GP to intelligently select the next point to evaluate:
1. Fit a GP to all previous observations
2. Use an acquisition function (Expected Improvement) to find the most promising next point
3. Evaluate that point (run the model)
4. Add the observation to the GP and repeat

Expected Improvement balances:
- **Exploitation**: Evaluating points where the GP predicts high performance
- **Exploration**: Evaluating points where the GP is uncertain

This makes Bayesian Optimization much more efficient than random search - it learns from previous evaluations to focus on promising regions.

### Evaluation

Each candidate configuration is evaluated by:
1. Wrapping the base model with `GroupedSteeringModelWrapper`
2. Applying steering vectors with the specified group weights
3. Running the model on the eval set
4. Computing pass@1 score

This is expensive (each eval takes time), which is why Bayesian Optimization is so valuable - it minimizes the number of evaluations needed.

## Results

![Bayesian Optimization 2D](figs/gp_optimization_2d.gif)

With Qwen 2.5 7B on MATH:
- **Baseline**: 33.75% pass@1
- **Finetuned**: 56% pass@1
- **Base + optimal steering**: ~56% pass@1

The best weights found were:
```json
{
  "early": 0.0,
  "mid_early": 1.0,
  "mid_late": 0.0,
  "late": 1.0
}
```

This basically matches full finetuning performance! The optimal weights reveal an interesting pattern: only the mid-early and late layer groups benefit from steering, while early and mid-late layers are best left alone.

The Gaussian Process visualizations show:
- How the optimization explores the search space
- Which regions are most promising
- How uncertainty decreases as we observe more points
- Whether there are multiple good solutions or one clear optimum

This approach gives us both practical benefits (finding good weights) and scientific insights (understanding which parts of the model benefit most from steering). The fact that we can optimize in a 4D space with relatively few evaluations demonstrates the power of Bayesian Optimization for expensive black-box problems.

