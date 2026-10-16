"""Qwen's questions in the Draft node's box (swap_draft.questions_text / question_overrides / edited_questions)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import conftest  # noqa: E402,F401  (ComfyUI stubs)
import swap_draft as sd  # noqa: E402


def test_default_box_changes_nothing():
    text = sd.questions_text()
    for name, const, _ in sd.QUESTIONS:
        assert f"### {name} (" in text
    assert sd.question_overrides(text) == ({}, [])
    assert sd.question_overrides("") == ({}, [])


def test_an_edit_applies_inside_and_is_restored():
    text = sd.questions_text().replace("Answer only \"yes\" or \"no\".", "Answer yes or no, nothing else.")
    over, warn = sd.question_overrides(text)
    assert list(over) == ["RECHECK_INSTRUCTION"] and not warn
    before = sd.RECHECK_INSTRUCTION
    with sd.edited_questions(over):
        assert sd.RECHECK_INSTRUCTION.endswith("Answer yes or no, nothing else.")
        assert sd.RECHECK_INSTRUCTION.format(who="x")
    assert sd.RECHECK_INSTRUCTION == before
    assert sd.question_overrides(sd.questions_text()) == ({}, [])     # still the built-in after


def test_bad_edits_are_ignored_with_a_warning():
    text = sd.questions_text().replace("The main person in this video is {who}.",
                                       "The main person in this video is {who} in {place}.")
    over, warn = sd.question_overrides(text)
    assert "MOMENT_INSTRUCTION" not in over and "{place}" in warn[0]
    over, warn = sd.question_overrides("### nonsense\nhello")
    assert not over and "unknown section" in warn[0]


def test_touch_edit_flows_into_the_moment_question():
    text = sd.questions_text().replace('touch: where the main person', 'touch: where exactly the main person')
    over, _ = sd.question_overrides(text)
    with sd.edited_questions(over):
        assert "touch: where exactly the main person" in sd.moment_instruction("a man", touch=True)


def test_release_memory_reports_before_and_after():
    r = sd.release_memory()
    assert len(r["vram_free"]) == 2 and len(r["ram"]) == 2


def test_free_memory_switch_is_appended_last():
    opt = sd.SeamStitchSwapDraft.INPUT_TYPES()["optional"]
    assert list(opt) == ["sheet", "free_memory_after", "qwen_questions", "sheet_2"]
    assert opt["free_memory_after"][1]["default"] is True


def test_a_box_saved_with_an_older_built_in_is_not_an_edit():
    old_box = sd.questions_text().replace(sd.SCENE_INSTRUCTION, sd._LEGACY_QUESTIONS["SCENE_INSTRUCTION"][0])
    assert old_box != sd.questions_text()
    assert sd.question_overrides(old_box) == ({}, [])
    assert "a crowd or audience, seating, screens, a scoreboard" in sd.SCENE_INSTRUCTION
