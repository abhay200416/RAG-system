import re
import unicodedata


MAX_QUERY_LENGTH = 4000

_GREETING_RESPONSES = {
    "hi": "Hi! What would you like to know?",
    "hi there": "Hi! What would you like to know?",
    "hello": "Hello! What would you like to know?",
    "hello there": "Hello! What would you like to know?",
    "hey": "Hey! What would you like to know?",
    "hey there": "Hey! What would you like to know?",
    "hiya": "Hi! What would you like to know?",
    "howdy": "Hello! What would you like to know?",
    "good morning": "Good morning! What would you like to know?",
    "good afternoon": "Good afternoon! What would you like to know?",
    "good evening": "Good evening! What would you like to know?",
    "how are you": "I'm here and ready to help. What would you like to know?",
    "how are you doing": "I'm here and ready to help. What would you like to know?",
    "whats up": "Hi! What would you like to know?",
    "thanks": "You're welcome! Let me know if you have another question.",
    "thank you": "You're welcome! Let me know if you have another question.",
}

_JAILBREAK_PATTERNS = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\s+"
        r"(?:(?:all|any|the|your|previous|prior|above|earlier|system|developer)\s+){0,5}"
        r"(?:instructions?|prompts?|rules?|polic(?:y|ies))\b"
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget|override)\s+"
        r"(?:(?:the|your)\s+)?(?:system|developer)\b"
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:everything|all)\s+"
        r"(?:you\s+were\s+told|you\s+have\s+been\s+told|prior\s+context)\b"
    ),
    re.compile(
        r"\b(?:replace|rewrite|change)\s+(?:your\s+)?"
        r"(?:system|developer)\s+(?:prompts?|instructions?|rules?)\b"
    ),
    re.compile(
        r"\b(?:bypass|disable|override|ignore)\s+"
        r"(?:(?:the|all|your)\s+)?(?:safety|guardrails?|filters?|polic(?:y|ies))\b"
    ),
    re.compile(
        r"\b(?:reveal|show|print|repeat|disclose|output|quote)\s+"
        r"(?:(?:the|your|hidden|system|developer)\s+){0,4}"
        r"(?:system\s+|developer\s+)?(?:prompts?|instructions?|messages?|rules?)\b"
    ),
    re.compile(
        r"\b(?:show|tell|give|reveal|print|repeat|quote|disclose)\s+"
        r"(?:me\s+)?(?:your|the|hidden|system|developer)\s+"
        r"(?:(?:hidden|system|developer)\s+)?"
        r"(?:prompts?|instructions?|messages?|rules?)\b"
    ),
    re.compile(
        r"\bwhat\s+(?:is|are)\s+(?:your|the)\s+"
        r"(?:(?:hidden|system|developer)\s+)*(?:prompts?|instructions?|messages?)\b"
    ),
    re.compile(
        r"\b(?:show|reveal|print|dump|give|extract|list|send|exfiltrate)\s+"
        r"(?:(?:me|the|your|all|hidden)\s+){0,4}"
        r"(?:api\s*keys?|passwords?|credentials?|access\s*tokens?|secrets?)\b"
    ),
    re.compile(
        r"\bwhat\s+(?:is|are)\s+(?:your|the)\s+"
        r"(?:api\s*keys?|passwords?|credentials?|access\s*tokens?|secrets?)\b"
    ),
    re.compile(
        r"\b(?:developer|dan|unrestricted|unfiltered)\s+mode\b|"
        r"\bdo\s+anything\s+now\b|"
        r"\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered)\b|"
        r"\byou\s+are\s+(?:now\s+)?(?:dan|unrestricted|unfiltered)\b|"
        r"\b(?:pretend|act)\s+(?:that\s+)?(?:you\s+)?"
        r"(?:have|follow)\s+no\s+(?:rules|restrictions|policies)\b"
    ),
)

_JAILBREAK_REFUSAL = (
    "I can't follow requests to override instructions, reveal hidden prompts or "
    "secrets, or bypass safety rules. Please ask a question about the retrieved "
    "information instead."
)


def validate_query(query: object) -> str:
    """Validate and normalize a user query before retrieval or generation."""
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    normalized = query.strip()
    if not normalized:
        raise ValueError("Query cannot be empty")
    if len(normalized) > MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query cannot exceed {MAX_QUERY_LENGTH} characters"
        )
    if any(ord(character) < 32 and character not in "\n\r\t" for character in normalized):
        raise ValueError("Query contains unsupported control characters")
    return normalized


def greeting_response(query: str) -> str | None:
    """Return a local response for a standalone greeting or thanks message."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    return _GREETING_RESPONSES.get(normalized)


def guardrail_response(query: str) -> str | None:
    """Return a refusal for common instruction-override and secret-extraction asks."""
    normalized = unicodedata.normalize("NFKC", query).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = " ".join(normalized.split())
    if any(pattern.search(normalized) for pattern in _JAILBREAK_PATTERNS):
        return _JAILBREAK_REFUSAL
    return None