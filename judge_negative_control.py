"""Negative-control evaluation: verify the LLM judge detects a known-wrong answer.

This is NOT a real ShopBot response; the control is explicitly labeled in Langfuse.
Runs one LLM judge call only (no ShopBot generation).
"""

from langfuse import observe, propagate_attributes
from ai_assistant_v2 import evaluate_answer, langfuse


@observe(name="judge-negative-control")
def run_negative_control():
    question = "Can I return a shirt I've worn after 45 days?"
    known_wrong_answer = (
        "Yes, you can return a worn shirt within 60 days, even without tags."
    )
    expected = (
        "Reject returns after 30 days or for worn items; tags must be attached."
    )

    with propagate_attributes(
        version="v2-negative-control",
        tags=["evaluation-v2", "negative-control", "llm-judge"],
        metadata={
            "test_type": "negative_control",
            "answer_source": "deliberately_incorrect_test_fixture",
            "note": "Not a real ShopBot answer",
        },
    ):
        verdict = evaluate_answer(question, known_wrong_answer, expected)
        for metric, reason_key in (
            ("policy_compliance", "policy_reason"),
            ("answer_relevance", "relevance_reason"),
        ):
            langfuse.score_current_trace(
                name=f"{metric}_llm_v2",
                value=float(verdict[metric]),
                data_type="NUMERIC",
                comment=f"NEGATIVE CONTROL — known incorrect test answer. {verdict[reason_key]}",
            )
        print("Negative control (NOT a real bot answer)")
        print("Question:", question)
        print("Answer:", known_wrong_answer)
        print("Policy compliance:", verdict["policy_compliance"])
        print("Answer relevance:", verdict["answer_relevance"])
        print("Policy reason:", verdict["policy_reason"])
        print("Relevance reason:", verdict["relevance_reason"])


if __name__ == "__main__":
    try:
        run_negative_control()
    finally:
        langfuse.flush()
