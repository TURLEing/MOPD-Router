# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IFEval-style instruction-following constraint verification.

This module is a self-contained port of allenai/open-instruct's
``open_instruct/if_functions.py`` (Apache-2.0), covering all 25 constraints
from the IFEval taxonomy. It is used by the MOPD-Router data pipeline to score
single-domain ``ifeval`` prompts and the synthesized ``math_if_composite``
prompts (math correctness AND instruction-following, both must hold).

The only third-party dependency is ``langdetect`` (used solely by
``validate_response_language``); the import is deferred so the module stays
importable when the package is absent. A failing/missing language detector
simply scores language constraints as 0.0.
"""

import inspect
import json
import re

try:
    import langdetect  # type: ignore

    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


# include keywords: Include keywords {keyword1}, {keyword2} in your response
def verify_keywords(text, keyword_list):
    response_lower = text.lower()
    return all(keyword.lower() in response_lower for keyword in keyword_list)


# Keyword Frequency: In your response, the word {word} should appear {N} times.
def verify_keyword_frequency(text, word, N):
    text = text.lower()
    keyword = word.lower()
    words = re.findall(r"\b\w+\b", text)
    actual_count = sum(1 for w in words if w == keyword)
    return actual_count == N


# Forbidden Words: Do not include keywords {forbidden words} in the response.
def validate_forbidden_words(text, forbidden_words):
    text_lower = text.lower()
    found_words = [word for word in forbidden_words if word.lower() in text_lower]
    return len(found_words) == 0


# Letter Frequency: In your response, the letter {letter} should appear {N} times.
def verify_letter_frequency(text, letter, N):
    if len(letter) != 1:
        return False
    return text.count(letter) == N


# Response Language: Your ENTIRE response should be in {language}, no other language is allowed.
def validate_response_language(text, language):
    if not _HAS_LANGDETECT:
        return False
    try:
        detected_language = langdetect.detect(text)
    except Exception:
        return False
    return detected_language == language


# Number Paragraphs (markdown divider): Your response should contain {N} paragraphs separated by '* * *'.
def verify_paragraph_count(text, N):
    def clean_text(t):
        return "\n".join(line.strip() for line in t.splitlines()).strip()

    text = clean_text(text)
    paragraphs = text.split("* * *")
    actual_count = len(paragraphs)
    valid_paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if len(valid_paragraphs) != actual_count:
        return False
    return actual_count == N


# Number Words: Answer with at least / around / at most {N} words
def validate_word_constraint(text, N, quantifier):
    words = text.strip().split()
    actual_count = len(words)
    tolerance = max(round(N * 0.1), 1)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "at most":
        return actual_count <= N
    elif quantifier == "around":
        return abs(actual_count - N) <= tolerance
    return False


# Number Sentences: Answer with at least / around / at most {N} sentences.
def verify_sentence_constraint(text, N, quantifier):
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)
    actual_count = len(sentences)
    if quantifier == "at least":
        return actual_count >= N
    elif quantifier == "around":
        return abs(actual_count - N) <= 1
    elif quantifier == "at most":
        return actual_count <= N
    return False


# Number Paragraphs + First Word: {N} paragraphs separated by two line breaks,
# the {i}-th paragraph must start with {first word}.
def validate_paragraphs(text, N, first_word, i):
    paragraphs = text.split("\n\n")
    if len(paragraphs) != N:
        return False
    return bool(paragraphs[i - 1].strip().startswith(first_word))


# Postscript: At the end of your response, add a postscript starting with {postscript marker}
def verify_postscript(text, postscript_marker):
    if postscript_marker in text:
        marker_index = text.find(postscript_marker)
        remaining_text = text[marker_index:].strip()
        return len(remaining_text) > len(postscript_marker)
    return False


# Number Placeholder: at least {N} placeholders in square brackets, e.g. [address].
def validate_placeholders(text, N):
    pattern = r"\[(.*?)\]"
    placeholders = re.findall(pattern, text)
    return len(placeholders) >= N


# Number Bullets: exactly {N} markdown bullet points.
def verify_bullet_points(text, N):
    lines = text.split("\n")
    bullet_points = [line.strip() for line in lines if line.strip().startswith(("*", "-"))]
    return len(bullet_points) == N


# Title: a title wrapped in double angular brackets, e.g. <<poem of joy>>.
def validate_title(text):
    pattern = r"<<(.*?)>>"
    return len(re.findall(pattern, text)) > 0


# Choose: Answer with one of the following options: {options}
def validate_choice(text, options):
    return any(option in text for option in options)


# Highlighted Section: at least {N} sections with markdown *highlighted section*
def validate_highlighted_sections(text, N):
    pattern = r"\*(.*?)\*"
    return len(re.findall(pattern, text)) >= N


# Multiple Sections: {N} sections marked with {section splitter} X.
def validate_sections(text, N, section_splitter):
    sections = text.split(section_splitter)
    if sections and sections[0] == "":
        sections.pop(0)
    return len(sections) == N


# JSON Format: entire output wrapped in JSON.
def validate_json_format(text):
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


# Repeat Prompt: first repeat the request without change, then answer.
def validate_repeat_prompt(text, original_prompt):
    return bool(text.startswith(original_prompt))


# Two Responses: two different responses separated by '******'.
def validate_two_responses(text):
    if text.count("******") == 1:
        response_list = text.split("******")
        first_response = response_list[0].strip()
        second_response = response_list[1].strip()
        if first_response != second_response:
            return True
    return False


# All Uppercase: entire response in English, capital letters only.
def validate_uppercase(text):
    return text == text.upper()


# All Lowercase: entire response in English, all lowercase.
def validate_lowercase(text):
    return text == text.lower()


# Frequency of All-capital Words: at least / around / at most {N} times.
def validate_frequency_capital_words(text, N, quantifier):
    words = re.findall(r"\b[A-Z]+\b", text)
    if quantifier == "at least":
        return len(words) >= N
    elif quantifier == "around":
        return abs(len(words) - N) <= max(round(N * 0.1), 1)
    elif quantifier == "at most":
        return len(words) <= N
    return False


# End Checker: finish with the exact phrase {end phrase}.
def validate_end(text, end_phrase):
    return bool(text.endswith(end_phrase))


# Quotation: wrap entire response with double quotation marks.
def validate_quotation(text):
    return bool(text.startswith('"') and text.endswith('"'))


# No Commas: no commas anywhere in the response.
def validate_no_commas(text):
    return "," not in text


IF_FUNCTIONS_MAP = {
    "verify_keywords": verify_keywords,
    "verify_keyword_frequency": verify_keyword_frequency,
    "validate_forbidden_words": validate_forbidden_words,
    "verify_letter_frequency": verify_letter_frequency,
    "validate_response_language": validate_response_language,
    "verify_paragraph_count": verify_paragraph_count,
    "validate_word_constraint": validate_word_constraint,
    "verify_sentence_constraint": verify_sentence_constraint,
    "validate_paragraphs": validate_paragraphs,
    "verify_postscript": verify_postscript,
    "validate_placeholders": validate_placeholders,
    "verify_bullet_points": verify_bullet_points,
    "validate_title": validate_title,
    "validate_choice": validate_choice,
    "validate_highlighted_sections": validate_highlighted_sections,
    "validate_sections": validate_sections,
    "validate_json_format": validate_json_format,
    "validate_repeat_prompt": validate_repeat_prompt,
    "validate_two_responses": validate_two_responses,
    "validate_uppercase": validate_uppercase,
    "validate_lowercase": validate_lowercase,
    "validate_frequency_capital_words": validate_frequency_capital_words,
    "validate_end": validate_end,
    "validate_quotation": validate_quotation,
    "validate_no_commas": validate_no_commas,
}

# Constraint types that are incompatible with a math final-answer expressed via
# \boxed{} (they either forbid the characters/structure that math requires, or
# force a shape that collides with it). Used to filter the math x IF composite
# builder so synthetic composite prompts remain satisfiable.
MATH_INCOMPATIBLE_CONSTRAINTS = {
    "All Lowercase",
    "All Uppercase",
    "No Commas",
    "JSON Format",
    "Quotation",
    "Two Responses",
    "Response Language",
}


def _coerce(value):
    """Best-effort coercion of JSON-decoded ground_truth values to validator arg types."""
    return value


def apply_constraint(text, constraint_spec):
    """Apply a single IFEval constraint to a response string.

    Args:
        text: the model response (decoded).
        constraint_spec: a dict carrying a ``func_name`` key plus the keyword
            arguments expected by that validator (extra/None keys are ignored).
            This is exactly the ``ground_truth`` JSON object stored in the
            ``allenai/RLVR-GSM-MATH-IF-Mixed-Constraints`` ``ifeval`` subset.

    Returns:
        True if the constraint holds, False otherwise (also False if the
        func_name is unknown or required args are missing).
    """
    if not isinstance(constraint_spec, dict):
        return False
    func_name = constraint_spec.get("func_name")
    func = IF_FUNCTIONS_MAP.get(func_name) if isinstance(func_name, str) else None
    if func is None:
        return False

    # Only keep kwargs the validator actually accepts, dropping None values so
    # validators with optional/positional params don't receive them.
    sig_params = set(inspect.signature(func).parameters)
    sig_params.discard("text")
    kwargs = {k: v for k, v in constraint_spec.items() if k in sig_params and v is not None}
    try:
        return bool(func(text, **kwargs))
    except Exception:
        return False


def compute_score(model_output: str, ground_truth, timeout_score: float = 0.0) -> float:
    """Score a response against an IFEval constraint spec.

    ``ground_truth`` is the JSON string stored in the dataset row (e.g.
    ``'{"func_name": "validate_lowercase", ...}'``). Returns 1.0 if the
    constraint is satisfied, else 0.0. Malformed specs also score 0.0.
    """
    try:
        spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    except (ValueError, TypeError):
        return timeout_score
    return 1.0 if apply_constraint(model_output, spec) else 0.0
