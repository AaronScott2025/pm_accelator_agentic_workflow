from __future__ import annotations

from ai_interviewer_pm.agents.schema import RubricScoreModel, parse_rubric_json


def test_rubric_parsing_valid() -> None:
    payload = {
        "clarity": 4,
        "structure": 4.5,
        "product_sense": 4,
        "metrics": 3,
        "stakeholder": 4,
        "prioritization": 3.5,
        "strategy": 4,
        "summary": "Solid response with measurable impact.",
    }
    m = RubricScoreModel(**payload)
    assert m.clarity == 4


def test_rubric_parsing_from_text() -> None:
    txt = (
        "Here is the score: {\n  \"clarity\": 3, \n  \"structure\": 3, \n  \"product_sense\": 3, \n"
        "  \"metrics\": 2, \n  \"stakeholder\": 3, \n  \"prioritization\": 3, \n  \"strategy\": 2, \n  \"summary\": \"OK\"\n}"
    )
    parsed = parse_rubric_json(txt)
    assert isinstance(parsed, dict)
    assert "clarity" in parsed
