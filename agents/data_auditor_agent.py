import pandas as pd
from agents.base_agent import BaseAgent

class DataAuditorAgent(BaseAgent):
    def __init__(self):
        super().__init__("DataAuditorAgent")

    def act(self, state):
        self.log("Auditing ratings dataset")

        df = pd.read_csv(state["ratings_path"])

        report = {
            "num_users": df.userId.nunique(),
            "num_items": df.movieId.nunique(),
            "num_ratings": len(df),
            "sparsity": 1 - (len(df) / (df.userId.nunique() * df.movieId.nunique()))
        }

        self.log(f"Audit report: {report}")
        state["data_report"] = report
        return state
