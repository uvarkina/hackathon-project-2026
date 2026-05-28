import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv
load_dotenv()


# Path to the phrases file
PHRASES_PATH = Path(__file__).parent / "fraud_phrases.json"

# Category mapping: each phrase index range maps to a category
# We'll detect category by checking which phrases matched
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

    # Return category with highest score, default to bank_fraud
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

    transcript_lower = transcript_text.lower()
    matched_phrases = []

    for phrase in phrases:
        if phrase.lower() in transcript_lower:
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
    Use OpenAI API to assess if the transcript is a fraudulent call.

    Returns:
        dict with ai_score (0.0 to 1.0) and ai_category
    """
    lang_name = "Hebrew" if language == "he" else "Russian"

    prompt = (
        f"You are a fraud detection expert specializing in phone scams in Israel.\n"
        f"Analyze the following {lang_name} phone call transcript and determine:\n"
        f"1. How likely is this a fraudulent/scam call? (score from 0.0 to 1.0)\n"
        f"2. If fraudulent, what category: bank_fraud, police_impersonation, "
        f"social_security, tax_authority, lottery_scam, or none\n\n"
        f"Transcript: \"{transcript_text}\"\n\n"
        f"Respond ONLY in valid JSON format:\n"
        f'{{\"fraud_score\": <float>, \"category\": \"<category>\"}}'
    )

    prompt = 'Answer ONLY in valid JSON format  {fraud_score: 10.0, category: \"bank_fraud\"}'

    try:
        print('--')
        client = openai.OpenAI()  # Uses OPENAI_API_KEY env variable
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        print(5)

        content = response.choices[0].message.content.strip()
        print(content, '--------')
        # Parse JSON from response
        result = json.loads(content)
        ai_score = float(result.get("fraud_score", 0.0))
        ai_category = result.get("category", "none")

        # Clamp score to [0, 1]
        ai_score = max(0.0, min(1.0, ai_score))

        return {"ai_score": ai_score, "ai_category": ai_category}

    except (openai.OpenAIError, json.JSONDecodeError, KeyError, ValueError):
        print('==')
        # If AI detection fails, return neutral score
        return {"ai_score": 0.0, "ai_category": "none"}


def check_fraud_phrases(transcript_text: str, language: str) -> dict:
    """
    Check if a transcript contains fraud indicators using two methods:
    1. Simple pattern matching against known fraud phrases
    2. AI-based analysis for contextual fraud detection

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

    # Method 2: AI-based detection
    ai_result = _ai_detection(transcript_text, language)

    # Combine scores: weighted average (pattern: 40%, AI: 60%)
    # Pattern matching is reliable but rigid; AI catches novel phrasings
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
    test_ru = "Здравствуйте, это полиция Израиля, вы подозреваетесь в отмывании денег. На ваше имя выписан ордер на арест. Три слона идут гулять. Это полиция израиля, вы подозреваетесь в отмывании."

    print("=== Hebrew Test ===")
    result_he = check_fraud_phrases(test_he, "he")
    print(json.dumps(result_he, ensure_ascii=False, indent=2))

    print("\n=== Russian Test ===")
    result_ru = check_fraud_phrases(test_ru, "ru")
    print(json.dumps(result_ru, ensure_ascii=False, indent=2))
