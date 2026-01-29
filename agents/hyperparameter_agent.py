# agents/hyperparameter_agent.py

import random

SEARCH_SPACE = {
    "factors": [32, 64, 128],
    "regularization": [0.01, 0.05, 0.1],
    "alpha": [20, 40]
}

def act(state: dict) -> dict:
    print("[HyperparameterAgent] Exploring hyperparameters...")

    iteration = state.get("iteration", 0)
    state["iteration"] = iteration + 1

    params = {
        "factors": random.choice(SEARCH_SPACE["factors"]),
        "regularization": random.choice(SEARCH_SPACE["regularization"]),
        "alpha": random.choice(SEARCH_SPACE["alpha"])
    }

    state["params"] = params

    print(f"[HyperparameterAgent] Selected: {params}")

    return state

