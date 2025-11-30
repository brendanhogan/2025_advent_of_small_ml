# 2025 Advent of Small ML 

![Advent of Small ML Header](figs/day0.png)

I have many little research ideas I've been curious to try and also a lot of research to catch up on (especially after being out on paternity leave!). Some are more research-oriented, some are more abstract/artistic/weird. 

I'd love to give more time to each one, but as a forcing function I wanted to do the advent of ML ideas, for 25 days until Christmas I want to put out a half-baked idea with some initial experimentation. (And 1-2 posts will just be catching up on my reading list.)

Each day will get a folder under the `src/` directory. Each day should have a README that describes the experiment, code, and results - but will also link to the Twitter post that gives more detail. 

## Self-Contained Experiments

**Each day/folder is totally self-contained.** Even if some days build off previous days, each folder will be a self-contained project,  even if it means repeated code. While all experiments share the same `uv` environment (defined by the root `uv.lock`), each day's code is independent and can be run on its own without dependencies on other days.

Yes, this means there will be repeated code across folders. But the goal here is simplicity and ease of experimentation - you can test things out, copy a folder, modify it, and not worry about breaking other experiments. 

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for package management and uses a shared environment across all experiments. The repo includes a `uv.lock` file, which allows you to recreate the environment identically.

To get started:

1. Install uv (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync the environment (creates virtual environment and installs exact pinned versions from `uv.lock`):
   ```bash
   uv sync
   ```

## Structure

Each day's experiment lives in `src/day_XX/` with its own README documenting the approach, code, and results.

---

Some (or maybe all) of these ideas might be worth fleshing out more - if you're interested in that or just want to talk more, please message me on Twitter [@brendanh0gan](https://x.com/brendanh0gan)



