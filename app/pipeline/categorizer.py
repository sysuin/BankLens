"""
NLP transaction categorizer for BankLens.

Maps raw transaction descriptions to human-readable spending categories
using a keyword dictionary. This approach is:

    - Fast: no external model, no API call, runs instantly
    - Transparent: the category assignment is fully explainable
    - Extensible: add new keywords by editing KEYWORD_MAP

Supported categories:
    Income        — salary, freelance payments, interest, refunds
    Food          — restaurants, food delivery, grocery stores
    Transport     — cabs, fuel, metro, flights, parking
    Utilities     — electricity, internet, gas, mobile bills
    Subscriptions — streaming, SaaS tools, memberships
    Shopping      — e-commerce, retail, apparel
    Health        — pharmacy, hospital, insurance, diagnostics
    Savings       — FD transfers, SIP, mutual funds, RD
    Others        — anything that does not match a keyword
"""

import pandas as pd

from app.core.logger import get_logger

logger = get_logger(__name__)

# ── Keyword Dictionary ────────────────────────────────────────────────────────
# Each key is a category name. Each value is a list of lowercase substrings
# to match against the transaction description.
#
# Matching is sequential: the first category whose keyword is found wins.
# Order within KEYWORD_MAP matters — 'Income' is checked first to correctly
# classify inflows before any debit-side pattern can match.

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
    ],
    "Utilities": [
        "electricity",
        "water bill",
        "gas bill",
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
    """
    Assign a spending category to a single transaction description.

    Normalises the description to lowercase and checks it against each
    category's keyword list in order. Returns the first matching category,
    or 'Others' if no keyword matches.

    Args:
        description: The raw transaction description string from the CSV.

    Returns:
        A category string, e.g. 'Food', 'Transport', 'Income', 'Others'.

    Examples:
        >>> categorize("Zomato Order #12345")
        'Food'
        >>> categorize("Salary Credit - March")
        'Income'
        >>> categorize("Unknown Merchant XYZ")
        'Others'
    """
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
    """
    Apply categorize() to every row of the transactions DataFrame.

    Adds a new 'category' column without modifying the original DataFrame.

    Args:
        df: A pandas DataFrame that must contain a 'description' column.

    Returns:
        A new DataFrame with an added 'category' column.
    """
    result = df.copy()
    result["category"] = result["description"].apply(categorize)
    logger.info("Categorized %d transactions.", len(result))
    return result
