import numpy as np
from .utils import sent_sim_cos

class TextTile:
    def __init__(self):
        pass

    def depth_scores(self, sim_scores):
        assert len(sim_scores) > 0
        sim_scores = np.array(sim_scores)
        left_peaks = sim_scores.copy()
        right_peaks = sim_scores.copy()
        offset = 0
        for i, score in enumerate(sim_scores[:-2]):
            if sim_scores[i+1] < score:
                right_peaks[offset:i+1] = score
                offset = i+1
        offset = len(sim_scores)
        for i in range(len(sim_scores)-1, 0, -1):
            score = sim_scores[i]
            if sim_scores[i-1] < score:
                left_peaks[i:offset] = score
                offset = i
        depths = left_peaks + right_peaks - 2*sim_scores
        return depths[:-1]

    def _align_boundary(self, predicted_boundary_ids, original_boundary_ids):
        for i, pid in enumerate(predicted_boundary_ids):
            # avoid exhausts original ids before all aligned
            preserve_to = len(original_boundary_ids) - len(predicted_boundary_ids) + i + 1
            aligned_oid_at = preserve_to - 1
            dist = original_boundary_ids[aligned_oid_at]
            for j, oid in enumerate(original_boundary_ids[:preserve_to]):
                dist0 = abs(pid-oid)
                if dist0 > dist: break
                dist, aligned_oid_at = dist0, j
            predicted_boundary_ids[i] = original_boundary_ids[aligned_oid_at]
            # avoid duplicating or even forward boundary, will change the list
            del original_boundary_ids[:aligned_oid_at+1]
        return predicted_boundary_ids

    def cut_paragraphs(self, sent_words, num_paras=None, block_sents=3, std_weight=0.5,
                       align_boundary=True, original_boundary_ids=None):

        sims = [0 for i in range(len(sent_words))]
        # for i in range(block_sents, len(sentences)-block_sents):
        for i in range(1, len(sent_words)):
            left_words = [x for words in sent_words[max(0, i-block_sents):i] for x in words]
            right_words = [x for words in sent_words[i:min(len(sent_words), i+block_sents)] for x in words]
            sims[i-1] = sent_sim_cos(left_words, right_words)

        depths = self.depth_scores(sims)    # ignore the last one, must be boundary

        if num_paras is None:   # automatically determine according to stats
            num_paras = np.sum(depths > np.mean(depths) - std_weight*np.std(depths))
            if align_boundary:
                assert original_boundary_ids is not None
                if num_paras >= len(original_boundary_ids):
                    return original_boundary_ids

        # last sentence must be a boundary
        predicted_boundary_ids = (1+np.argsort(depths)[::-1][:num_paras-1]).tolist() + [len(sent_words)]
        predicted_boundary_ids = list(sorted(predicted_boundary_ids))
        if align_boundary:
            # move the predicted boundary to the nearest original one to align
            # if 2 predicted boundaries falls in the same original paragraph, move it to the next one
            predicted_boundary_ids = self._align_boundary(predicted_boundary_ids, original_boundary_ids)
        return predicted_boundary_ids
