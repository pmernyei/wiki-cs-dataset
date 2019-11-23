import numpy as np

class WikiDataNode:
    def __init__(self, id, title, label, outlinks, tokens):
        self.id = id
        self.title = title
        self.outlinks = outlinks
        self.tokens = tokens
        self.label = label
        self.vector = np.array([])
