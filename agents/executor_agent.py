from agents.planner_agent import PlannerAgent
from agents.data_auditor_agent import DataAuditorAgent
from agents.model_selector_agent import ModelSelectorAgent
from agents.trainer_agent import TrainerAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.memory_agent import MemoryAgent

class ExecutorAgent:
    def __init__(self, ratings_path):
        self.state = {
            "ratings_path": ratings_path
        }

        self.agents = {
            "audit_data": DataAuditorAgent(),
            "select_model": ModelSelectorAgent(),
            "train_model": TrainerAgent(),
            "evaluate_model": EvaluatorAgent(),
            "store_memory": MemoryAgent()
        }

        self.planner = PlannerAgent()

    def execute(self):
        print("\n===== STARTING AGENTIC SYSTEM =====\n")

        self.state = self.planner.act(self.state)

        for step in self.state["plan"]:
            self.state = self.agents[step].act(self.state)

        print("\n===== FINAL STATE =====")
        print(self.state)
