"""
NLP transaction categorizer for BankLens.

Maps raw transaction descriptions to human-readable spending categories
using a keyword dictionary. This approach is:

    - Fast: no external model, no API call, runs instantly
    - Transparent: the category assignment is fully explainable
    - Extensible: add new keywords by editing KEYWORD_MAP

Supported categories:
    Income         — salary, freelance payments, interest, refunds
    Rent & Housing — house rent, apartment maintenance, society fees
    Education      — school fees, college tuition, coaching
    Food           — restaurants, food delivery, grocery stores
    Transport      — cabs, fuel, metro, flights, car repair
    Utilities      — electricity, internet, gas cylinder, mobile bills
    Subscriptions  — streaming, SaaS tools, memberships
    Health         — pharmacy, hospital, insurance, diagnostics
    Shopping       — e-commerce, retail, apparel
    Savings        — FD transfers, SIP, mutual funds, RD
    Others         — anything that does not match a keyword
"""

import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Keyword Dictionary ────────────────────────────────────────────────────────
KEYWORD_MAP: dict[str, list[str]] = {
    "Income": [
        "salary",
        "credit",
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
        "auto",
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
    "Savings": [
        "savings transfer",
        "fixed deposit",
        "mutual fund",
        "sip",
        "ppf",
        "nps",
        "recurring deposit",
        "transfer to savings",
        "fd opening",
        "rd installment",
        "investment",
    ],
}


def categorize(description: str) -> str:
    """Assign a spending category to a single transaction description."""
    normalized = description.lower().strip()

    for category, keywords in KEYWORD_MAP.items():
        for keyword in keywords:
            if keyword in normalized:
                logger.debug(
                    "Matched '%s' → %s (keyword: '%s')",
                    description,
                    category,
                    keyword,
                )
                return category

    logger.debug("No keyword match for '%s' → Others", description)
    return "Others"


def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply categorize() to every row of the transactions DataFrame."""
    result = df.copy()
    result["category"] = result["description"].apply(categorize)
    logger.info("Categorized %d transactions.", len(result))
    return result
