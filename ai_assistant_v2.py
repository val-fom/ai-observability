"""ShopBot + Langfuse: v2 traces and LLM-as-a-Judge evaluation.

Based on val-fom/ai-observability (2026-11-18).
Requires the existing .env with OpenAI and Langfuse keys.
Optional: JUDGE_MODEL=gpt-4.1-mini (enabled for the OpenAI project).
"""
import json
import os

from dotenv import load_dotenv
from langfuse import Langfuse, observe, propagate_attributes
from langfuse.openai import openai

from shared.shopbot_config import SHOPBOT_SYSTEM_PROMPT

load_dotenv(".env", override=True)

client = openai.OpenAI()
langfuse = Langfuse()
MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

# Retained from v1 so that we can compare keyword scoring to model-based scoring
# on the original return-policy question.
EXPECTED_FACTS = {
    "What is your return policy?": ["30", "unworn", "tag"],
}

EVALUATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "shopbot_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "policy_compliance": {"type": "number", "enum": [0, 0.5, 1]},
                "answer_relevance": {"type": "number", "enum": [0, 0.5, 1]},
                "policy_reason": {"type": "string"},
                "relevance_reason": {"type": "string"},
            },
            "required": [
                "policy_compliance", "answer_relevance",
                "policy_reason", "relevance_reason",
            ],
            "additionalProperties": False,
        },
    },
}


@observe(name="policy-judge-v2")
def evaluate_answer(question: str, answer: str, expected_behavior: str) -> dict:
    """Independent second API call, evaluated against the authoritative policy."""
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluator, NOT the shop assistant. "
                    "Use ONLY the supplied store policy as ground truth. "
                    "The expected behavior is a test criterion, not a substitute "
                    "for the policy. Do not infer missing store details. "
                    "Grade each metric using exactly 0, 0.5, or 1. "
                    "policy_compliance: 1 = fully consistent and no invented "
                    "policy facts; 0.5 = ambiguous, incomplete, or partially "
                    "unsupported; 0 = contradicts policy or fabricates a "
                    "material fact. A correct keyword in an incorrect claim "
                    "does not earn points. "
                    "answer_relevance: 1 = directly addresses the question "
                    "(including honestly saying something is unknown); "
                    "0.5 = partly addresses it; 0 = unrelated or evasive. "
                    "Write brief evidence-based reasons for both metrics. "
                    "Return JSON only, matching the required schema."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "store_policy": SHOPBOT_SYSTEM_PROMPT,
                        "question": question,
                        "expected_behavior": expected_behavior,
                        "shopbot_answer": answer,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=EVALUATION_SCHEMA,
        max_completion_tokens=350,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Judge returned empty content; check finish_reason/token limit")
    return json.loads(content)


@observe(name="chatbot-v2")
def ask_bot(question: str, expected_behavior: str) -> str:
    with propagate_attributes(
        user_id="homework-demo",
        session_id="homework-eval-v2",
        version="v2-llm-judge",
        tags=["shopbot", "evaluation-v2", "llm-judge"],
        metadata={
            "evaluation_version": "v2",
            "evaluation_method": "llm-as-judge",
            "judge_model": JUDGE_MODEL,
        },
    ):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SHOPBOT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content or ""

        # The old baseline score applies only to this exact original question.
        if question in EXPECTED_FACTS:
            expected = EXPECTED_FACTS[question]
            keyword_score = sum(word in answer.lower() for word in expected) / len(expected)
            langfuse.score_current_trace(
                name="policy_fact_coverage_v1",
                value=float(keyword_score),
                data_type="NUMERIC",
                comment="Baseline keyword coverage, not semantic correctness",
            )
            print(f"Keyword score (v1): {keyword_score:.2f}")

        verdict = evaluate_answer(question, answer, expected_behavior)
        langfuse.score_current_trace(
            name="policy_compliance_llm_v2",
            value=float(verdict["policy_compliance"]),
            data_type="NUMERIC",
            comment=verdict["policy_reason"],
        )
        langfuse.score_current_trace(
            name="answer_relevance_llm_v2",
            value=float(verdict["answer_relevance"]),
            data_type="NUMERIC",
            comment=verdict["relevance_reason"],
        )
        print(f"Q: {question}\nA: {answer}")
        print(f"Policy score: {verdict['policy_compliance']} — {verdict['policy_reason']}")
        print(f"Relevance score: {verdict['answer_relevance']} — {verdict['relevance_reason']}\n")
        return answer


TEST_CASES = [
    (
        "What is your return policy?",
        "State the 30-day return window, unworn requirement and attached tags.",
    ),
    (
        "Can I return a shirt I've worn after 45 days?",
        "Explain it does not meet the unworn condition or 30-day window.",
    ),
    (
        "Do you ship to Ukraine? How long does delivery take?",
        "State shipping availability and delivery times are not specified, "
        "without inventing a yes/no policy or timing.",
    ),
]


if __name__ == "__main__":
    print(f"Generating with {MODEL}; judging with {JUDGE_MODEL}")
    try:
        for question, expected_behavior in TEST_CASES:
            ask_bot(question, expected_behavior)
    finally:
        langfuse.flush()
