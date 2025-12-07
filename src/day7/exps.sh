
#
uv run python main.py --output_dir runs/baseline_current_reward --reward_mode current --num_train_iters 500 --eval_every 50 --save_every 1000

# 2) Entropy reward only
uv run python main.py --output_dir runs/entropy_only_reward --reward_mode entropy_only --num_train_iters 500  --eval_every 50 --save_every 1000

# 3) Combined rewards (current + entropy)
uv run python main.py --output_dir runs/combined_rewards_0.1 --reward_mode combined --num_train_iters 500 --eval_every 50 --save_every 1000

uv run python main.py --output_dir runs/combined_rewards_0.1 --reward_mode combined --num_train_iters 500 --eval_every 50 --save_every 1000

uv run python main.py --output_dir runs/combined_rewards_correct_only_0.1 --reward_mode combined --only_if_correct --num_train_iters 500 --eval_every 50 --save_every 1000 