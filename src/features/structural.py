import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")

def avg_sentence_length(text: str) -> float:
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return 0.0
    lengths = [len(sent.text.split()) for sent in sents]
    return sum(lengths) / len(lengths)

def avg_noun_phrases_per_sentence(text: str) -> float:
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return 0.0
    counts = [sum(1 for nc in doc.noun_chunks if nc.root.sent == sent)
              for sent in sents]
    return sum(counts) / len(counts)

def avg_verb_count_per_sentence(text: str) -> float:
    doc = nlp(text)
    sents = list(doc.sents)
    if not sents:
        return 0.0
    counts = [sum(1 for tok in sent if tok.pos_ == "VERB")
              for sent in sents]
    return sum(counts) / len(counts)

def proper_noun_ratio(text: str) -> float:
    doc = nlp(text)
    tokens = list(doc)
    if not tokens:
        return 0.0
    return sum(1 for tok in tokens if tok.pos_ == "PROPN") / len(tokens)

def extract_structural(text: str) -> np.ndarray:
    return np.array([
        avg_sentence_length(text),
        avg_noun_phrases_per_sentence(text),
        avg_verb_count_per_sentence(text),
        proper_noun_ratio(text),
    ], dtype=float)