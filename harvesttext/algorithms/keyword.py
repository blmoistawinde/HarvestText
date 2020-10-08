import numpy as np
import networkx as nx

def combine(word_list, window = 2):
    """构造在window下的单词组合，用来构造单词之间的边。
    
    :params word_list: list of str, 由单词组成的列表。
    :params window: int, 窗口大小。
    """
    if window < 2: window = 2
    for x in range(1, window):
        if x >= len(word_list):
            break
        word_list2 = word_list[x:]
        res = zip(word_list, word_list2)
        for r in res:
            yield r

def textrank(block_words, topK, with_score=False, window=2, weighted=False):
    G = nx.Graph()
    for word_list in block_words:
        for u, v in combine(word_list, window):
            if not weighted:
                G.add_edge(u, v)
            else:
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)

    pr = nx.pagerank_scipy(G)
    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    if with_score:
        return pr_sorted[:topK]
    else:
        return [w for (w, imp) in pr_sorted[:topK]]