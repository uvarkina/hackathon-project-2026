import json
from pathlib import Path
from difflib import SequenceMatcher

# transformers НЕ импортируется здесь — только внутри _get_classifier().
# Импорт на уровне модуля грузит C++ расширения PyTorch при старте uvicorn,
# что на macOS вызывает "mutex lock failed" и краш Python.


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

def _normalize(text: str) -> str:
    """Нормализация: строчные + ё→е (Whisper часто пишет е вместо ё)."""
    return text.lower().replace("ё", "е").replace("Ё", "е")

def is_similar(str1, str2, threshold1=0.6):
    similarity = SequenceMatcher(None, str1, str2).ratio()
    return similarity > threshold1

def _phrase_matches(phrase: str, transcript: str) -> bool:
    """
    Проверяет входит ли фраза в транскрипт.
    1. Точное вхождение (после нормализации)
    2. Пословное: ≥60% ключевых слов фразы (len>3) найдены в транскрипте
       по первым 5 символам — чтобы "переведите" совпало с "перевести".
    """
    p = _normalize(phrase)
    t = _normalize(transcript)

    if p in t:
        return True

    p_words = p.split()
    key_words = [w for w in p_words if len(w) > 3]
    if not key_words:
        return False

    matched = sum(1 for w in key_words if w[:5] in t)
    return matched / len(key_words) >= 0.6

def _get_classifier():
    """Lazy-load a multilingual zero-shot classification model (runs locally, free)."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline   # lazy — не при старте сервиса
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

    matched_phrases = []

    for phrase in phrases:
        if _phrase_matches(phrase, transcript_text):
            matched_phrases.append(phrase)

    # Score растёт с каждой новой совпавшей фразой
    n = len(matched_phrases)
    if n == 0:   pattern_score = 0.0
    elif n == 1: pattern_score = 0.51
    elif n == 2: pattern_score = 0.68
    elif n == 3: pattern_score = 0.84
    elif n == 4: pattern_score = 0.96
    else:        pattern_score = min(0.96 + (n - 4) * 0.04, 0.97)

    category = _detect_category(matched_phrases, language) if matched_phrases else "none"

    return {
        "pattern_score": pattern_score,
        "matched_phrases": matched_phrases,
        "category": category,
    }


def _ai_detection(transcript_text: str, language: str) -> dict:
    """
    Zero-shot mDeBERTa отключён: грузит ~1 ГБ модель и считает 1-3 сек
    на каждый 3-секундный чанк → текст вылазит с задержкой 5+ секунд.
    Pattern matching на ивритских/русских фразах работает достаточно хорошо.
    Чтобы включить обратно — убрать ранний return ниже.
    """
    return {"ai_score": 0.0, "ai_category": "none"}

    # ── disabled below ──
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

    # Без PyTorch ai_score = 0, поэтому score = pattern_score напрямую
    combined_score = round(pattern_score + 0.4 * ai_score, 2)
    combined_score = min(combined_score, 0.97)

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
