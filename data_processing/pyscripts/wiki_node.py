import numpy as np

class WikiDataNode:
    """
    Represent extracted data about a single node in the Wikipedia graph (i.e. a
    single article): page ID, title, outgoing links to other nodes in dataset,
    tokens in article text, class label. Also vectorized feature representation
    to be cocatenated by appropriate functions.
    """
    def __init__(self, id, title, label, outlinks, tokens):
        self.id = id
        self.title = title
        self.outlinks = outlinks
        self.tokens = tokens
        self.label = label
        self.vector = np.array([])
