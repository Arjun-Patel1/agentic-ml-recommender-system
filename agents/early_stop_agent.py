class EarlyStopAgent:
    def act(self, state):
        max_rejects = 3

        rejects = state.get("reject_count", 0)

        if not state.get("accepted", False):
            rejects += 1
        else:
            rejects = 0

        state["reject_count"] = rejects

        if rejects >= max_rejects:
            print("[EarlyStopAgent] Early stopping triggered 🛑")
            state["stop"] = True
        else:
            state["stop"] = False

        return state
