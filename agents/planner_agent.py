from agents.base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__("PlannerAgent")

    def act(self, state):
        self.log("Creating execution plan")

        plan = [
            "audit_data",
            "select_model",
            "train_model",
            "evaluate_model",
            "store_memory"
        ]

        state["plan"] = plan
        return state



