# agents/evaluator_agent.py

from metrics.recall import recall_at_k

def act(state: dict) -> dict:
    print("[EvaluatorAgent] Evaluating model...")

    model = state["model"]
    interaction_matrix = state["interaction_matrix"]

    recall = recall_at_k(
        model,
        interaction_matrix,
        k=10,
        max_users=500
    )

    state["recall@10"] = recall

    print(f"[EvaluatorAgent] Recall@10 = {recall:.4f}")

    return state
