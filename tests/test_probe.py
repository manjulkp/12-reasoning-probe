import json
from fractions import Fraction

import pytest

from probe import (
    DISTRACTOR,
    add_distractor,
    boxed_contents,
    build_prompt,
    compare_pairs,
    extract_answer,
    gold_answer,
    is_correct,
    load_problems,
    parse_number,
    summarize,
    write_report,
)


class TemplateTokenizer:
    def __init__(self, ending):
        self.ending = ending

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return f"<user>{messages[0]['content']}<assistant>{self.ending}"


def test_gold_answer():
    assert gold_answer("She has 2+2=4 apples\n#### 4") == "4"
    assert gold_answer("Total\n#### 1,200") == "1,200"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("42", Fraction(42)),
        ("1,200", Fraction(1200)),
        ("$3.50", Fraction(7, 2)),
        ("-5", Fraction(-5)),
        ("1/2", Fraction(1, 2)),
        ("about 18 dollars", Fraction(18)),
        ("no digits", None),
        ("1/0", None),
    ],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_boxed_uses_last_and_handles_nesting():
    assert boxed_contents("\\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert boxed_contents("none") is None
    assert boxed_contents("\\boxed{unclosed") is None


def test_answer_comes_from_after_thinking():
    assert extract_answer("<think>maybe 5</think>The answer is \\boxed{7}") == 7
    assert extract_answer("<think>maybe 5</think>So 3 plus 4 gives 7.") == 7


def test_unfinished_thinking_has_no_answer():
    assert extract_answer("<think>maybe \\boxed{5} and so on") is None


def test_response_without_thinking_can_still_answer():
    assert extract_answer("The answer is \\boxed{7}") == 7


def test_is_correct_compares_numeric_values():
    assert is_correct("\\boxed{1200}", "1,200")
    assert is_correct("\\boxed{18.00}", "18")
    assert not is_correct("\\boxed{19}", "18")
    assert not is_correct("no number at all", "18")


def test_distractor_goes_before_the_final_question():
    result = add_distractor("Sam has 3 apples. Ann has 2. How many in total?")
    assert result == f"Sam has 3 apples. Ann has 2. {DISTRACTOR} How many in total?"


def test_distractor_on_single_sentence():
    assert add_distractor("What is 2+2?") == f"{DISTRACTOR} What is 2+2?"


@pytest.mark.parametrize("ending", ["", "<think>\n"])
def test_prompts_open_exactly_one_think_block(ending):
    tokenizer = TemplateTokenizer(ending)
    thinking = build_prompt(tokenizer, "Q?", "thinking")
    assert thinking.count("<think>") == 1
    assert thinking.endswith("<think>\n")
    no_thinking = build_prompt(tokenizer, "Q?", "no_thinking")
    assert no_thinking.count("<think>") == 1
    assert no_thinking.endswith("</think>\n\n")


def test_distractor_condition_adds_the_sentence_and_others_do_not():
    tokenizer = TemplateTokenizer("")
    question = "Sam has 3. How many?"
    assert DISTRACTOR in build_prompt(tokenizer, question, "distractor")
    assert DISTRACTOR not in build_prompt(tokenizer, question, "thinking")


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="unknown condition"):
        build_prompt(TemplateTokenizer(""), "Q?", "shouting")


def test_problem_sample_is_stable_and_uses_cache(tmp_path):
    path = tmp_path / "gsm.json"
    path.write_text(json.dumps([{"question": str(i), "answer": str(i)} for i in range(50)]))
    first = load_problems(10, path)
    assert first == load_problems(10, path)
    assert len(first) == 10
    assert first[:5] == load_problems(5, path)


RECORDS = [
    {"problem": 0, "condition": "thinking", "correct": True, "truncated": False, "tokens": 300, "seconds": 60},
    {"problem": 1, "condition": "thinking", "correct": True, "truncated": True, "tokens": 500, "seconds": 100},
    {"problem": 0, "condition": "no_thinking", "correct": False, "truncated": False, "tokens": 10, "seconds": 2},
    {"problem": 1, "condition": "no_thinking", "correct": True, "truncated": False, "tokens": 30, "seconds": 4},
]


def test_summarize_groups_by_condition():
    rows = {row["condition"]: row for row in summarize(RECORDS)}
    assert rows["thinking"]["correct"] == 2
    assert rows["thinking"]["truncated"] == 1
    assert rows["thinking"]["tokens"] == 400
    assert rows["no_thinking"]["correct"] == 1
    assert "distractor" not in rows


def test_compare_pairs():
    assert compare_pairs(RECORDS, "thinking", "no_thinking") == {"pairs": 2, "only_first": 1, "only_second": 0}


def test_report_is_written(tmp_path):
    report = write_report(RECORDS, tmp_path / "results.md")
    assert "2/2" in report and "1/2" in report
    assert (tmp_path / "results.md").read_text(encoding="utf-8") == report + "\n"
