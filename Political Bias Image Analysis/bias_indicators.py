import re

# These are research indicators, not definitive political classifications.
# A phrase may appear in support, criticism, quotation, or neutral reporting.

INDICATORS = {
    "liberal-associated": [
        "reproductive rights",
        "abortion rights",
        "climate justice",
        "gun control",
        "universal healthcare",
        "healthcare for all",
        "living wage",
        "workers' rights",
        "labor rights",
        "racial justice",
        "systemic racism",
        "student loan forgiveness",
        "wealth inequality",
        "renewable energy",
        " LGBTQ rights",
        "immigrant rights",
        "voting rights",
        "social safety net",
        "criminal justice reform",
        "environmental protection",
    ],
    "conservative-associated": [
        "border security",
        "illegal immigration",
        "tax cuts",
        "lower taxes",
        "limited government",
        "second amendment",
        "gun rights",
        "school choice",
        "parental rights",
        "traditional values",
        "law and order",
        "religious liberty",
        "fiscal responsibility",
        "energy independence",
        "deregulation",
        "national sovereignty",
        "military strength",
        "constitutional originalism",
        "sanctity of life",
        "free market",
    ],
}

EMOTIONAL_TERMS = {
    "positive": [
        "hope", "progress", "success", "freedom", "opportunity",
        "justice", "fairness", "unity", "protect", "support",
        "improve", "victory", "peace",
    ],
    "negative": [
        "crisis", "threat", "danger", "corrupt", "failure",
        "disaster", "fear", "destroy", "attack", "betray",
        "injustice", "cruel", "shame", "dangerous",
    ],
}

NEGATIONS = {
    "not", "never", "no", "against", "oppose", "opposes",
    "opposed", "reject", "rejects", "without"
}


def analyze_caption(caption):
    text = str(caption or "")
    normalized = text.lower()
    matches = []

    for category, phrases in INDICATORS.items():
        for phrase in phrases:
            pattern = r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)"

            for match in re.finditer(pattern, normalized):
                start = match.start()
                preceding = normalized[max(0, start - 65):start]
                words_before = re.findall(r"\b[\w']+\b", preceding)
                nearby_words = words_before[-6:]

                negated = any(word in NEGATIONS for word in nearby_words)
                quoted = (
                    normalized.rfind('"', 0, start) >
                    normalized.rfind('"', 0, start - 1)
                )

                matches.append({
                    "category": category,
                    "phrase": phrase.strip(),
                    "evidence": text[match.start():match.end()],
                    "negation_nearby": negated,
                    "quotation_context_possible": quoted,
                })

    # Count language markers, not political lean.
    emotional_counts = {}
    for label, terms in EMOTIONAL_TERMS.items():
        found = [
            term for term in terms
            if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", normalized)
        ]
        emotional_counts[label] = found

    hashtags = re.findall(r"#[\w]+", text)
    mentions = re.findall(r"@[\w.]+", text)

    return {
        "caption_length": len(text),
        "word_count": len(text.split()),
        "indicators": matches,
        "emotional_language": emotional_counts,
        "hashtags": hashtags,
        "mentions": mentions,
        "disclaimer": (
            "Phrase matches are contextual research indicators, not proof "
            "of endorsement or the author's political beliefs. Check "
            "negation, quotation, sarcasm, and surrounding context."
        ),
    }
