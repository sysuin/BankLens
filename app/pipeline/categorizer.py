"""
NLP transaction categorizer for BankLens.

Maps raw transaction descriptions to human-readable spending categories
using a 2-Stage Hybrid Approach:

    - Stage 1 (Rules): Fast keyword matching against KEYWORD_MAP (instant, 0 cost).
    - Stage 2 (LLM Fallback): Batch LLM classification for any items assigned "Others",
      eliminating the fragile fallback bucket and handling regional/messy merchant names.

Supported categories:
    Income, Rent & Housing, Education, Food, Transport, Utilities,
    Subscriptions, Health, Shopping, Savings, Others
"""

import json
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Keyword Dictionary (Order-independent matching with strict phrases) ────────
KEYWORD_MAP: dict[str, list[str]] = {
    "Savings": [
        "fixed deposit",
        "fd opening",
        "sip equity",
        "sip mutual",
        "mutual fund",
        "ppf deposit",
        "provident fund",
        "recurring deposit",
        "savings transfer",
        "investment",
        "nps",
        "fd auto transfer",
        "auto transfer to fd",
        "auto transfer",
    ],
    "Income": [
        "salary",
        "inflow credit",
        "freelance",
        "payment received",
        "transfer in",
        "interest earned",
        "refund",
        "cashback",
        "bonus",
        "incentive",
        "dividend",
        "reimbursement",
    ],
    "Rent & Housing": [
        "rent",
        "house maintenance",
        "housing",
        "lease",
        "apartment",
        "society fee",
        "society maintenance",
        "landlord",
    ],
    "Education": [
        "school",
        "college",
        "tuition",
        "coaching",
        "fees",
        "education",
        "course",
        "academy",
    ],
    "Food": [
        "zomato",
        "swiggy",
        "uber eats",
        "dominos",
        "pizza hut",
        "kfc",
        "mcdonalds",
        "restaurant",
        "cafe",
        "coffee",
        "grocery",
        "supermarket",
        "metro mart",
        "bigbasket",
        "blinkit",
        "dunzo",
        "zepto",
        "food",
        "bakery",
        "juice",
    ],
    "Transport": [
        "ola",
        "uber",
        "rapido",
        "metro rail",
        "railway",
        "irctc",
        "petrol",
        "fuel",
        "parking",
        "toll",
        "cab",
        "auto ride",
        "auto cab",
        "auto rickshaw",
        "bus",
        "flight",
        "airline",
        "indigo",
        "air india",
        "makemytrip",
        "goibibo",
        "redbus",
        "car repair",
        "service station",
        "mechanic",
        "workshop service",
        "repair",
    ],
    "Utilities": [
        "electricity",
        "water bill",
        "gas bill",
        "gas cylinder",
        "cylinder",
        "lpg",
        "broadband",
        "wifi",
        "airtel",
        "jio",
        "bsnl",
        "vi ",
        "vodafone",
        "dish tv",
        "tata sky",
        "recharge",
        "mobile bill",
        "postpaid",
        "internet bill",
        "utility",
    ],
    "Subscriptions": [
        "netflix",
        "amazon prime",
        "hotstar",
        "disney",
        "spotify",
        "youtube premium",
        "apple",
        "microsoft",
        "adobe",
        "notion",
        "canva",
        "github",
        "subscription",
        "membership",
        "autocad",
    ],
    "Health": [
        "pharmacy",
        "medical",
        "hospital",
        "clinic",
        "doctor",
        "apollo",
        "medplus",
        "healthkart",
        "health insurance",
        "medicine",
        "lab",
        "diagnostic",
        "nursing",
        "dental",
        "chemist",
    ],
    "Shopping": [
        "amazon",
        "flipkart",
        "myntra",
        "ajio",
        "nykaa",
        "meesho",
        "snapdeal",
        "reliance digital",
        "croma",
        "mall",
        "retail",
        "fashion",
        "footwear",
        "apparel",
    ],
}


def categorize(description: str) -> str:
    """Assign a spending category to a single transaction description via rule matching."""
    normalized = str(description).lower().strip()

    for category, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in normalized:
                return category

    return "Others"


def batch_llm_categorize_others(
    uncategorized_descriptions: list[str],
) -> dict[str, str]:
    """
    Stage 2: Batch LLM classification for items assigned 'Others'.
    Returns a dictionary mapping description -> category.
    """
    if not uncategorized_descriptions or not settings.openai_api_key:
        return {desc: "Others" for desc in uncategorized_descriptions}

    try:
        categories_list = list(KEYWORD_MAP.keys()) + ["Others"]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a financial transaction classification engine. Classify each merchant/description into one of these exact categories: "
                    f"{categories_list}. Output JSON format mapping description to category. Return JSON only.",
                ),
                ("human", "Classify these transaction descriptions: {descriptions}"),
            ]
        )
        llm = ChatOpenAI(
            model=settings.openai_mini_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )
        chain = prompt | llm
        res = chain.invoke({"descriptions": json.dumps(uncategorized_descriptions)})
        content = res.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        logger.info("Stage 2 LLM categorizer classified %d transactions.", len(parsed))
        return parsed
    except Exception as e:
        logger.warning("Stage 2 LLM categorization failed, using default: %s", e)
        return {desc: "Others" for desc in uncategorized_descriptions}


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply 2-Stage Hybrid Categorization to every row of the transactions DataFrame."""
    result = df.copy()

    # Stage 1: Rule-based matching
    result["category"] = result["description"].apply(categorize)

    # Stage 2: LLM Fallback for 'Others'
    others_mask = result["category"] == "Others"
    others_descs = result.loc[others_mask, "description"].unique().tolist()

    if others_descs and settings.openai_api_key:
        llm_mapped = batch_llm_categorize_others(others_descs)
        result.loc[others_mask, "category"] = result.loc[
            others_mask, "description"
        ].map(lambda d: llm_mapped.get(d, "Others"))

    logger.info("Categorized %d transactions (2-stage hybrid).", len(result))
    return result
