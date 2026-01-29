from agents.llm_reasoner import reason
from agents.memory_agent import MemoryAgent

class ModelSelectorAgent:
    def __init__(self):
        self.memory = MemoryAgent()

    def act(self, state: dict):
        print("[ModelSelectorAgent] Reasoning with memory...")

        past_best = self.memory.best_model()

        prompt = f"""
You are an ML recommender expert.

Past best experiment:
{past_best}

Current task:
- Dataset: MovieLens ratings
- Goal: maximize Recall@10

Choose model: als or bpr
Respond with only model name.
"""

        model = reason(prompt).strip().lower()

        if model not in ["als", "bpr"]:
            model = "als"

        state["model_name"] = model
        return state
