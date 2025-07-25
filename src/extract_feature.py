from features.sequence import extract_sequence
from features.structural import extract_structural
from features.stylometry import extract_stylometry

import numpy as np
import pandas as pd

def read_extract(df:pd.DataFrame):
    text_features = []
    for t in df["text"]:
        sty = extract_stylometry(t)
        stru = extract_structural(t)
        seq = extract_sequence(t)

        feats = np.concatenate([sty, stru, seq])
        text_features.append(feats)
    X = np.stack(text_features)
    Y = df["label"].to_numpy()
    return X, Y