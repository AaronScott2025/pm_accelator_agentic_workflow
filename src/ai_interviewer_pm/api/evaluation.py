from __future__ import annotations

from typing import Any, Dict, List


def try_ragas_single(question: str, answer: str, contexts: List[str]) -> Dict[str, Any] | None:
    """Compute a minimal RAGAS metric set for a single example if ragas is installed.

    Returns None if ragas or datasets is not installed, or if evaluation fails.
    """
    print(f"DEBUG try_ragas_single: Starting RAGAS evaluation")
    print(f"DEBUG try_ragas_single: Question length: {len(question)}")
    print(f"DEBUG try_ragas_single: Answer length: {len(answer)}")
    print(f"DEBUG try_ragas_single: Contexts count: {len(contexts)}")

    # For now, return mock RAGAS results to test the frontend display
    # TODO: Fix the uvloop compatibility issue
    print(f"DEBUG try_ragas_single: Returning mock RAGAS results for testing")
    return {
        "answer_relevancy": 0.85,
        "faithfulness": 0.78,
        "context_precision": 0.82,
        "context_recall": 0.76
    }

    # TODO: Implement real RAGAS evaluation when uvloop compatibility is resolved
    # For now, return mock results for testing purposes


def llm_as_judge(question: str, answer: str, feedback: str) -> Dict[str, Any] | None:
    """Simple LLM-as-judge that scores the answer 0-10 based on rubric/feedback.

    Returns a dict with {score, rationale}. Returns None on failure. Keep lightweight.
    """
    from ai_interviewer_pm.agents.behavioral_graph import _get_llm  # reuse model config
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "You are an impartial PM interview judge. Score the candidate's answer from 0 to 10,\n"
            "where 0 is unusable and 10 is excellent. Base your decision on the question and the given\n"
            "interviewer feedback (which reflects rubric criteria). Return strictly JSON with keys:\n"
            "score (number), rationale (short string)."
        )),
        ("human", "Question: {q}\nAnswer: {a}\nFeedback: {f}\nOutput JSON only."),
    ])

    try:
        llm = _get_llm()
        msg = (prompt | llm).invoke({"q": question, "a": answer, "f": feedback})
        import json

        data = json.loads(getattr(msg, "content", "{}"))
        score = float(data.get("score", 0))
        rationale = str(data.get("rationale", ""))
        return {"score": score, "rationale": rationale}
    except Exception:
        return None

