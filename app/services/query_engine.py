# query_engine.py
class QueryEngine:
    def __init__(self, index):
        self.query_engine = index.as_query_engine(
            streaming=True,
            similarity_top_k=3  # Adjust this value based on your needs
        )
    def query(self, question):
        response = self.query_engine.query(question)
        return str(response)

