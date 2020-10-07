import numpy as np
from collections import defaultdict

class EntRetrieveMixin:
    """
    实体检索模块:
    - 基于倒排索引快速检索包括某个实体的文档，以及统计出现某实体的文档数目
    """
    def build_index(self, docs, with_entity=True, with_type=True):
        inv_index = defaultdict(set)
        for i, sent in enumerate(docs):
            entities_info = self.entity_linking(sent)
            for span, (entity, type0) in entities_info:
                if with_entity:
                    inv_index[entity].add(i)
                if with_type:
                    inv_index[type0].add(i)
        return inv_index

    def get_entity_counts(self, docs, inv_index, used_type=[]):
        if len(used_type) > 0:
            entities = iter(x for x in self.entity_type_dict
                            if self.entity_type_dict[x] in used_type)
        else:
            entities = self.entity_type_dict.keys()
        cnt = {enty: len(inv_index[enty]) for enty in entities if enty in inv_index}
        return cnt

    def search_entity(self, query, docs, inv_index):
        words = query.split()
        if len(words) > 0:
            ids = inv_index[words[0]]
            for word in words[1:]:
                ids = ids & inv_index[word]
            np_docs = np.array(docs)[list(ids)]
            return np_docs.tolist()
        else:
            return []