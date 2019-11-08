class WikiDataNode:
    def __init__(self, id, title, labels, outlinks, tokens):
        self.id = id
        self.title = title
        self.outlinks = outlinks
        self.tokens = tokens
        self.labels = labels
