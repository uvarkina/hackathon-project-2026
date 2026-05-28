import json
from pathlib import Path
from difflib import SequenceMatcher

from transformers import pipeline


# Path to the phrases file
PHRASES_PATH = Path(__file__).parent / "fraud_phrases.json"

# Category mapping: each phrase index range maps to a category
CATEGORIES = [
    "bank_fraud",
    "police_impersonation",
    "social_security",
    "tax_authority",
    "lottery_scam",
]

# Keywords that hint at each category (used to classify matched phrases)
CATEGORY_KEYWORDS = {
    "bank_fraud": {
        "he": ["בנק", "חשבון", "כרטיס", "סיסמה", "עסקה", "קוד אימות", "ביטחון הבנק"],
        "ru": ["банк", "счёт", "карт", "пароль", "транзакц", "код подтверждения", "безопасности банка"],
    },
    "police_impersonation": {
        "he": ["משטרה", "מעצר", "קצין", "ניידת", "הלבנת הון", "חוקר"],
        "ru": ["полиц", "арест", "офицер", "наряд", "отмывани", "расследован"],
    },
    "social_security": {
        "he": ["ביטוח לאומי", "קצבה", "זכאות", "מענק"],
        "ru": ["Битуах Леуми", "пособи", "права", "грант", "компенсац"],
    },
    "tax_authority": {
        "he": ["מס הכנסה", "רשות המסים", "עיקול", "החזר מס"],
        "ru": ["налог", "Мас Ахнаса", "налоговая", "арест"],
    },
    "lottery_scam": {
        "he": ["זכית", "הגרלה", "פרס", "לוטו", "משלוח"],
        "ru": ["выиграли", "лотере", "приз", "лото", "доставк", "путёвк"],
    },
}

# Lazy-loaded classifier
_classifier = None

def is_similar(str1, str2, threshold1=0.6):
    # ratio() returns a float between 0.0 and 1.0
    similarity = SequenceMatcher(None, str1, str2).ratio()
    return similarity > threshold1

def _get_classifier():
    """Lazy-load a multilingual zero-shot classification model (runs locally, free)."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
            device=-1,  # CPU
        )
    return _classifier


def _load_phrases() -> dict:
    """Load fraud phrases from JSON file."""
    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_category(matched_phrases: list[str], language: str) -> str:
    """Determine the fraud category based on matched phrases."""
    category_scores = {cat: 0 for cat in CATEGORIES}

    for phrase in matched_phrases:
        phrase_lower = phrase.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            lang_keywords = keywords.get(language, [])
            for keyword in lang_keywords:
                if keyword.lower() in phrase_lower:
                    category_scores[category] += 1
                    break

    best_category = max(category_scores, key=category_scores.get)
    if category_scores[best_category] == 0:
        return "bank_fraud"
    return best_category


def _pattern_detection(transcript_text: str, language: str) -> dict:
    """
    Simple pattern matching against known fraud phrases.

    Returns:
        dict with pattern_score, matched_phrases, and category
    """
    phrases_data = _load_phrases()
    phrases = phrases_data.get(language, [])

    transcript_lower = transcript_text.lower().split()
    print(transcript_lower)
    matched_phrases = []

    for phrase in phrases:
        for forbidden_phares in transcript_lower:
            if is_similar(phrase, forbidden_phares):
        # if phrase.lower() in transcript_lower:
                matched_phrases.append(phrase)

    # Score: 0 for no match, 0.6 for 1 match, 0.85 for 2+ matches
    if len(matched_phrases) == 0:
        pattern_score = 0.0
    elif len(matched_phrases) == 1:
        pattern_score = 0.6
    else:
        pattern_score = 0.85

    category = _detect_category(matched_phrases, language) if matched_phrases else "none"

    return {
        "pattern_score": pattern_score,
        "matched_phrases": matched_phrases,
        "category": category,
    }


def _ai_detection(transcript_text: str, language: str) -> dict:
    """
    Use a free local zero-shot classification model to assess fraud likelihood.

    Uses MoritzLaurer/mDeBERTa-v3-base-mnli-xnli which supports
    multilingual text (100+ languages including Hebrew and Russian).
    Runs entirely locally — no API key or internet needed after first download.

    Returns:
        dict with ai_score (0.0 to 1.0) and ai_category
    """
    candidate_labels = [
        "bank fraud phone scam",
        "police impersonation scam",
        "social security benefits scam",
        "tax authority scam",
        "lottery prize scam",
        "normal legitimate phone call",
    ]

    category_map = {
        "bank fraud phone scam": "bank_fraud",
        "police impersonation scam": "police_impersonation",
        "social security benefits scam": "social_security",
        "tax authority scam": "tax_authority",
        "lottery prize scam": "lottery_scam",
        "normal legitimate phone call": "none",
    }

    try:
        classifier = _get_classifier()
        result = classifier(transcript_text, candidate_labels, multi_label=False)

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        ai_category = category_map.get(top_label, "none")

        # If top label is "normal legitimate phone call", invert the score
        if ai_category == "none":
            ai_score = 1.0 - top_score
        else:
            ai_score = top_score

        ai_score = round(max(0.0, min(1.0, ai_score)), 2)

        return {"ai_score": ai_score, "ai_category": ai_category}

    except Exception:
        # If AI detection fails, return neutral score
        return {"ai_score": 0.0, "ai_category": "none"}


def check_fraud_phrases(transcript_text: str, language: str) -> dict:
    """
    Check if a transcript contains fraud indicators using two methods:
    1. Simple pattern matching against known fraud phrases
    2. AI-based zero-shot classification (local, free)

    Combines both scores into a final text_score.

    Args:
        transcript_text: The transcribed text to analyze
        language: Language code ("he" or "ru")

    Returns:
        dict with:
            - text_score: Combined fraud score (0.0 to 1.0)
            - matched_phrases: List of matched fraud phrases
            - category: Detected fraud category
            - pattern_score: Score from pattern matching alone
            - ai_score: Score from AI analysis alone
    """
    # Method 1: Simple pattern detection
    pattern_result = _pattern_detection(transcript_text, language)

    # Method 2: AI-based detection (local model, no API key needed)
    ai_result = _ai_detection(transcript_text, language)

    # Combine scores: weighted average (pattern: 40%, AI: 60%)
    pattern_score = pattern_result["pattern_score"]
    ai_score = ai_result["ai_score"]

    combined_score = round(0.4 * pattern_score + 0.6 * ai_score, 2)

    # If pattern matched strongly, ensure minimum score
    if pattern_score >= 0.85:
        combined_score = max(combined_score, 0.75)
    elif pattern_score >= 0.6:
        combined_score = max(combined_score, 0.5)

    # Determine final category (prefer pattern category if phrases matched)
    if pattern_result["matched_phrases"]:
        category = pattern_result["category"]
    elif ai_result["ai_category"] != "none":
        category = ai_result["ai_category"]
    else:
        category = "none"

    return {
        "text_score": combined_score,
        "matched_phrases": pattern_result["matched_phrases"],
        "category": category,
        "pattern_score": pattern_score,
        "ai_score": ai_score,
    }


if __name__ == "__main__":
    # Example usage
    test_he = "שלום, אני מהבנק. החשבון שלך נפרץ, אנחנו צריכים לאמת את הפרטים שלך. אנא אשר את מספר הכרטיס שלך כדי לחסום את העסקה"
    test_ru = "Здравствуйте, это полиция Израиля, вы подозреваетесь в отмывании денег. На ваше имя выписан ордер на арест."

    print("=== Hebrew Test ===")
    result_he = check_fraud_phrases(test_he, "he")
    print(json.dumps(result_he, ensure_ascii=False, indent=2))

    print("\n=== Russian Test ===")
    result_ru = check_fraud_phrases(test_ru, "ru")
    print(json.dumps(result_ru, ensure_ascii=False, indent=2))
