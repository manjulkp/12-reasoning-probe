"""Probe a small reasoning model on grade-school maths (GSM8K).

The same model answers the same problems under three conditions:

- thinking: it reasons inside <think> tags before it answers, as designed.
- no thinking: its answer is forced to start right after an empty <think> block.
- distractor: it thinks, but the problem includes an irrelevant sentence.

Results are saved after each answer, so an interrupted run can continue where it stopped.
"""

import argparse
import json
import random
import re
import statistics
import time
from fractions import Fraction
from pathlib import Path

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DISTRACTOR = "Note that some of the items mentioned were slightly older than average."
CONDITIONS = ("thinking", "no_thinking", "distractor")
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def gold_answer(answer_text):
    """Read the final answer from GSM8K's '#### 42' format."""
    return answer_text.split("####")[-1].strip()


def parse_number(text):
    """Convert text such as '1,200', '3.50' or '1/2' to an exact Fraction, or None."""
    match = NUMBER.search(text)
    if not match:
        return None
    try:
        return Fraction(match.group().replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def boxed_contents(text):
    """Return the contents of the last \\boxed{...}, allowing nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    index = start + len("\\boxed{")
    depth, begin = 1, index
    while index < len(text):
        depth += {"{": 1, "}": -1}.get(text[index], 0)
        if depth == 0:
            return text[begin:index]
        index += 1
    return None


def extract_answer(response):
    """Find the model's answer in the text after thinking ends.

    Prefer \\boxed{}, otherwise take the last number. A response that stopped before
    finishing its thinking has no answer, because numbers there are only working.
    """
    if "<think>" in response and "</think>" not in response:
        return None
    final = response.split("</think>")[-1]
    boxed = boxed_contents(final)
    if boxed is not None:
        return parse_number(boxed)
    numbers = NUMBER.findall(final)
    return parse_number(numbers[-1]) if numbers else None


def is_correct(response, gold):
    answer = extract_answer(response)
    return answer is not None and answer == parse_number(gold)


def add_distractor(question):
    """Insert the distractor before the final sentence, which holds the actual question."""
    sentences = re.split(r"(?<=[.!?])\s+", question.strip())
    if len(sentences) < 2:
        return f"{DISTRACTOR} {question}"
    return " ".join([*sentences[:-1], DISTRACTOR, sentences[-1]])


def build_prompt(tokenizer, question, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "distractor":
        question = add_distractor(question)
    messages = [{"role": "user", "content": f"{question}\n{INSTRUCTION}"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    # Some chat templates open the think block themselves, so check before adding one.
    if not prompt.rstrip().endswith("<think>"):
        prompt = prompt + "<think>\n"
    if condition == "no_thinking":
        prompt = prompt + "\n</think>\n\n"
    return prompt


def load_problems(count, path, seed=0):
    """Load a fixed random sample of GSM8K's test problems, using a cached copy if present."""
    path = Path(path)
    if path.exists():
        problems = json.loads(path.read_text(encoding="utf-8"))
    else:
        from datasets import load_dataset

        problems = [
            {"question": item["question"], "answer": gold_answer(item["answer"])}
            for item in load_dataset("openai/gsm8k", "main", split="test")
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(problems), encoding="utf-8")
    random.Random(seed).shuffle(problems)
    return problems[:count]


def summarize(records):
    rows = []
    for condition in CONDITIONS:
        group = [r for r in records if r["condition"] == condition]
        if group:
            rows.append(
                {
                    "condition": condition,
                    "n": len(group),
                    "correct": sum(r["correct"] for r in group),
                    "truncated": sum(r["truncated"] for r in group),
                    "tokens": statistics.mean(r["tokens"] for r in group),
                    "seconds": statistics.mean(r["seconds"] for r in group),
                }
            )
    return rows


def compare_pairs(records, first, second):
    """Count problems that only one of two conditions got right."""
    by_problem = {}
    for record in records:
        by_problem.setdefault(record["problem"], {})[record["condition"]] = record["correct"]
    pairs = [p for p in by_problem.values() if first in p and second in p]
    return {
        "pairs": len(pairs),
        "only_first": sum(p[first] and not p[second] for p in pairs),
        "only_second": sum(p[second] and not p[first] for p in pairs),
    }


NAMES = {"thinking": "Thinking", "no_thinking": "No thinking", "distractor": "Thinking, with a distractor sentence"}


def write_report(records, path):
    lines = [
        "# Reasoning probe results",
        "",
        "| Condition | Correct | Accuracy | Ran out of tokens | Average tokens | Average seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summarize(records):
        lines.append(
            f"| {NAMES[row['condition']]} | {row['correct']}/{row['n']} | {row['correct'] / row['n']:.0%} | "
            f"{row['truncated']} | {row['tokens']:.0f} | {row['seconds']:.0f} |"
        )
    lines += ["", "## Same problems, compared in pairs", ""]
    for first, second in (("thinking", "no_thinking"), ("thinking", "distractor")):
        result = compare_pairs(records, first, second)
        if result["pairs"]:
            lines.append(
                f"- {NAMES[first]} vs {NAMES[second].lower()}: {result['only_first']} problems only the first got right, "
                f"{result['only_second']} only the second got right, out of {result['pairs']}."
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "answers.jsonl"
    records = []
    if log_path.exists():
        with log_path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
    done = {(r["problem"], r["condition"]) for r in records}

    problems = load_problems(args.problems, "data/gsm8k_test.json")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    print(f"Loaded {args.model}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

    for condition in args.conditions:
        for index, problem in enumerate(problems):
            if (index, condition) in done:
                continue
            prompt = build_prompt(tokenizer, problem["question"], condition)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            started = time.perf_counter()
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
            seconds = time.perf_counter() - started
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            # The prompt opened the think block, so restore it for the answer parser.
            full = ("<think>\n" if condition != "no_thinking" else "") + response
            record = {
                "problem": index,
                "condition": condition,
                "gold": problem["answer"],
                "correct": is_correct(full, problem["answer"]),
                "truncated": len(new_tokens) >= args.max_tokens,
                "tokens": len(new_tokens),
                "seconds": seconds,
                "response": response,
            }
            records.append(record)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(
                f"{condition:<12} {index + 1:>3}/{len(problems)}  {'right' if record['correct'] else 'wrong'}  "
                f"{len(new_tokens):>4} tokens  {seconds:.0f}s",
                flush=True,
            )
            write_report(records, out / "results.md")
    print(write_report(records, out / "results.md"))


if __name__ == "__main__":
    main()
