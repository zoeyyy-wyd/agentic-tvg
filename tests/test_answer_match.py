from agentic_tvg.answer_match import (
    ParsedAnswer,
    answer_matches,
    expand_aliases,
    normalize,
    parse_answer_qa,
)


def test_normalize():
    assert normalize("A red flag.") == "red flag"
    assert normalize("The  Right foot!") == "right foot"


def test_aliases_paren():
    a = expand_aliases("A dark gray stone (rock).")
    assert "dark gray stone rock" in a and "dark gray stone" in a and "rock" in a


def test_aliases_numbers():
    a = expand_aliases("Two.")
    assert "two" in a and "2" in a


def test_containment_word_boundary():
    assert answer_matches("he waves a red flag", ["red flag"])
    assert not answer_matches("bored", ["red"])          # substring != word match
    assert answer_matches("Red.", ["red"])


def test_paraphrase_via_alias():
    aliases = expand_aliases("A dark gray stone (rock).")
    assert answer_matches("a rock", aliases)             # the paren alias saves it


def test_enumeration_hack_blocked():
    # shortest alias "red" = 1 word; cap = 5 words; enumeration exceeds it
    assert not answer_matches("red orange yellow blue green purple", ["red"])
    # and even a within-cap answer must contain the alias contiguously
    assert not answer_matches("orange or maybe blue", ["red"])


def test_parse_answer_qa():
    good = "<think>saw it</think><answer>A red flag.</answer>"
    p = parse_answer_qa(good)
    assert p == ParsedAnswer(answer="A red flag.", format_ok=True)
    assert not parse_answer_qa("<answer>x</answer> trailing").format_ok
    assert not parse_answer_qa("<think>t</think><answer>a</answer><answer>b</answer>").format_ok
    assert parse_answer_qa("no tags").answer is None
