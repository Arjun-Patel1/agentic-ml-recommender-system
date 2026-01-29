# agents/critic_agent.py

from metrics.recall import recall_at_k

def act(state: dict) -> dict:
    print("[CriticAgent] Evaluating Recall@10...")

    model = state["model"]
    interaction_matrix = state["interaction_matrix"]

    recall = recall_at_k(
        model,
        interaction_matrix,
        k=10,
        max_users=500  # safety for speed
    )

    print(f"[CriticAgent] Recall@10 = {recall:.4f}")

    best = state.get("best_recall", 0.0)

    if recall > best:
        state["best_recall"] = recall
        state["improved"] = True
        print("[CriticAgent] Improvement found ✅")
    else:
        state["improved"] = False
        print("[CriticAgent] No improvement ❌")

    return state
