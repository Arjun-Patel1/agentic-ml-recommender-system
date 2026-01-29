# agents/explanation_agent.py

def act(state: dict) -> dict:
    params = state["params"]
    recall = state.get("best_recall", 0.0)

    explanation = f"""
The ALS recommender was trained using:
• Factors: {params['factors']}
• Regularization: {params['regularization']}
• Alpha: {params['alpha']}

Best Recall@10 achieved: {recall:.4f}
"""

    print("[ExplanationAgent] Generating explanation...")
    print(explanation)

    state["explanation"] = explanation
    return state
