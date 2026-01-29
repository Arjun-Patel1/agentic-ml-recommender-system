class InMemoryStore:
    def __init__(self):
        self.history = []

    def add(self, item):
        self.history.append(item)

    def fetch_all(self):
        return self.history
