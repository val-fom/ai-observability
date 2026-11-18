import os
import random
import sys
from dotenv import load_dotenv
from langfuse.openai import openai
from langfuse import observe, propagate_attributes
from langfuse import Langfuse



# Add the repo root to path so we can import from shared/
sys.path.append(os.path.abspath(".."))
from shared.shopbot_config import SHOPBOT_SYSTEM_PROMPT

load_dotenv(".env",override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
langfuse = Langfuse()

LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-5-mini"

print("Setup complete — Langfuse client initialised")
print(f"Traces will appear at: {LANGFUSE_BASE_URL}")

EXPECTED_FACTS = {
    "What is your return policy?": ["30", "unworn", "tag"],
    "What is your refund method?": ["original", "5", "7"],
    "What are your customer support hours?": ["monday", "friday", "9", "6"],
    "Do you offer free shipping?": ["75"],
}

@observe(name="chatbot")
def ask_bot(user_message: str) -> str:
    properties = {"feature": "bot", "channel": "cli"}

    # Attach metadata to the observed function.
    langfuse.update_current_span(metadata=properties)

    # Propagate standard IDs, tags, and metadata to nested observations.
    with propagate_attributes(
        user_id="user_" + str(random.randint(1000, 9999)),
        session_id="session_" + str(random.randint(1000, 9999)),
        tags=["shopbot", "support"],
        metadata=properties,
    ):
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SHOPBOT_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    answer = response.choices[0].message.content or ""

    expected = EXPECTED_FACTS[user_message]
    score = sum(
        fact in answer.lower() for fact in expected
    ) / len(expected)

    langfuse.score_current_trace(
        name="policy_fact_coverage",
        value=float(score),
        data_type="NUMERIC",
        comment="Automated keyword-based policy fact coverage",
    )

    return answer

ask_bot("What is your return policy?")
ask_bot("What is your refund method?")
ask_bot("What are your customer support hours?")
ask_bot("Do you offer free shipping?")
