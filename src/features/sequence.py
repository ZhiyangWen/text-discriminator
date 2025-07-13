import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

_TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2-xl")
_MODEL     = GPT2LMHeadModel.from_pretrained("gpt2-xl").eval()
_DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    _MODEL.cuda()
_TOKENIZER.pad_token = _TOKENIZER.eos_token

def _local_diff(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    return sum(abs(probs[i+1] - probs[i]) for i in range(len(probs)-1)) / (len(probs)-1)

def _local_diff2(probs: list[float]) -> float:
    if len(probs) < 2:
        return 0.0
    return sum((probs[i+1] - probs[i])**2 for i in range(len(probs)-1)) / (len(probs)-1)

def _compute_uid_surp(text: str):
    corpus = _TOKENIZER.eos_token + text
    enc = _TOKENIZER(corpus, return_tensors="pt", padding=False)
    input_ids = enc.input_ids[0].to(_DEVICE)
    mask      = enc.attention_mask[0].to(_DEVICE)

    with torch.no_grad():
        outputs = _MODEL(input_ids=input_ids, attention_mask=mask, labels=input_ids)
        logits  = outputs.logits

    seq_len    = mask.sum().item()
    surprisals = []
    total      = 0.0

    for i in range(seq_len - 1):
        probs = F.softmax(logits[i], dim=-1)
        tokid = input_ids[i+1]
        s     = -torch.log(probs[tokid]).item()
        surprisals.append(s)
        total += s

    mean_s = total / (seq_len - 1) if seq_len > 1 else 0.0
    var_s  = float(np.mean([(x - mean_s)**2 for x in surprisals]))
    return var_s, mean_s, surprisals

def extract_sequence(text: str) -> np.ndarray:
    uid_var, mean_s, surps = _compute_uid_surp(text)
    return np.array([
        uid_var,
        _local_diff(surps),
        _local_diff2(surps),
        mean_s
    ], dtype=float)