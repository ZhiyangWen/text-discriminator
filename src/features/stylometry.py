import spacy
import textstat#compute readability score
import numpy as np
from nltk.corpus import stopwords


nlp = spacy.load("en_core_web_sm")


import nltk
nltk.download("stopwords", quiet=True)
_STOPWORDS = set(stopwords.words("english"))

#Length/phrasing
def word_count(text: str) -> int:
    tokens = text.split()
    return len(tokens)

def sentence_count(text: str) -> int:
    doc = nlp(text)
    return len(list(doc.sents))

# Lexical richness
def type_token_ratio(text: str) -> float:
    tokens = text.split()
    return len(set(tokens)) / len(tokens) if tokens else 0.0

def hapax_legomenon_rate(text: str) -> float:
    tokens = text.split()
    return sum(1 for t in set(tokens) if tokens.count(t) == 1) / len(tokens) if tokens else 0.0

#Punctuation & stop‐words
def punctuation_ratio(text: str) -> float:
    punct_marks = set("?!,.;:'\"")
    total_chars = len(text)
    if total_chars == 0:
        return 0.0
    return sum(1 for c in text if c in punct_marks) / total_chars

def stop_word_ratio(text: str) -> float:
    tokens = text.lower().split()
    return sum(1 for w in tokens if w in _STOPWORDS) / len(tokens) if tokens else 0.0

#Readability
def readability(text: str) -> float:
    return textstat.flesch_reading_ease(text) if text else 0.0


def extract_stylometry(text: str) -> np.ndarray:
    return np.array([
        word_count(text),
        sentence_count(text),
        type_token_ratio(text),
        hapax_legomenon_rate(text),
        punctuation_ratio(text),
        stop_word_ratio(text),
        readability(text),
    ], dtype=float)