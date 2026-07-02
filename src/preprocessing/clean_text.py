

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def lowercase(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub("", text)


def remove_mentions(text: str) -> str:
    return MENTION_PATTERN.sub("", text)


def dehashtag(text: str) -> str:
    """Strip the '#' but keep the word, since hashtag words often carry sentiment."""
    return HASHTAG_PATTERN.sub(r"\1", text)


def remove_non_alpha(text: str) -> str:
    return NON_ALPHA_PATTERN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Full cleaning pipeline applied in order."""
    text = lowercase(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = dehashtag(text)
    text = remove_non_alpha(text)
    text = collapse_whitespace(text)
    return text


if __name__ == "__main__":
    sample = "@user Check this out! https://example.com #GreatDay soooo happy!!"
    print(clean_text(sample))
