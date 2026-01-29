# agents/governor_agent.py

def act(state: dict) -> dict:
    """
    Governor decides whether to continue or stop the graph.
    """

    iteration = state.get("iteration", 0)
    improved = state.get("improved", False)

    max_iters = state.get("max_iterations", 5)

    if improved:
        print("[GovernorAgent] Improvement detected → continue")
        state["stop"] = False
    elif iteration >= max_iters:
        print("[GovernorAgent] Max iterations reached → stop")
        state["stop"] = True
    else:
        print("[GovernorAgent] No improvement → continue search")
        state["stop"] = False

    return state
