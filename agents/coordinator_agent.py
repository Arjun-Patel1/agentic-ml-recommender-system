class CoordinatorAgent:
    """
    Controls stopping condition.
    """

    def act(self, state: dict) -> dict:
        if state["iteration"] >= 5:
            print("🛑 Stopping optimization")
            return {"stop": True}

        return {}
