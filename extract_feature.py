from features.sequence import extract_sequence
from features.structural import extract_structural
from features.stylometry import extract_stylometry

import numpy as np
import pandas as pd

def read_extract(df:pd.DataFrame):
    text = df["text"].tolist()
    label = df["label"].to_numpy(dtype=int)
    text_features = []
    for t in text:
        sty = extract_stylometry(t)
        stru = extract_structural(t)
        seq = extract_sequence(t)
        text_features.append([sty, stru, seq])
    