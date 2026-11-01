# Project 12: Reasoning-model probe

Tests how much a small reasoning model gains from thinking, using grade-school maths
problems from GSM8K. The model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, which is
small enough to try on a CPU and has an MIT license.

The same model answers the same 30 problems under three conditions:

| Condition | What happens |
| --- | --- |
| Thinking | It reasons inside `<think>` tags before answering, as designed |
| No thinking | Its answer is forced to start right after an empty `<think>` block |
| Distractor | It thinks, but the problem includes an irrelevant sentence |

An answer that runs out of tokens before thinking ends counts as wrong, and the report
shows how often that happens.

## Results

**Not available yet.** This experiment hasn't started. When it has run, results will be
in `outputs\results.md`, with every answer in `outputs\answers.jsonl`.

## Before you run it

- **Memory:** the model's weights alone need about 6 GB, so close other apps first.
- **Download:** the model is downloaded the first time it runs.
- **Time:** expect this to be slow on a CPU. Each answer can run up to 600 tokens.

## Run it

```powershell
uv sync
uv run python probe.py
uv run pytest
```

Progress is saved after every answer. If a run stops, run the same command again and it
continues from where it stopped.

| Option | Default | Meaning |
| --- | --- | --- |
| `--problems` | `30` | Problems to test |
| `--max-tokens` | `600` | Longest allowed answer |
| `--conditions` | All three | `thinking`, `no_thinking` and `distractor` |

With only 30 problems, small accuracy differences could be down to chance. The report
compares conditions problem by problem, which is more useful here than the overall
percentages.

## Files

| File | What it does |
| --- | --- |
| `probe.py` | Builds prompts, runs the model, checks answers and writes the report |
| `tests/` | Tests answer checking and prompts without loading the model |
