import numpy as np
from collections import Counter

def sent_sim_cos(words1, words2):
    eps = 1e-5
    bow1 = Counter(words1)
    norm1 = sum(x ** 2 for x in bow1.values()) ** 0.5 + eps
    bow2 = Counter(words2)
    norm2 = sum(x ** 2 for x in bow2.values()) ** 0.5 + eps
    cos_sim = sum(bow1[w] * bow2[w] for w in set(bow1) & set(bow2)) / (norm1 * norm2)
    return cos_sim

def sent_sim_textrank(words1, words2):
    if len(words1) <= 1 or len(words2) <= 1:
        return 0.0
    return (len(set(words1) & set(words2))) / (np.log2(len(words1)) + np.log2(len(words2)))

