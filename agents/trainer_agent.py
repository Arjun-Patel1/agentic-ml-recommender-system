# agents/trainer_agent.py

from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

def act(state: dict) -> dict:
    print("[TrainerAgent] Training model...")

    interaction_matrix = state["interaction_matrix"]
    params = state["params"]

    if not isinstance(interaction_matrix, csr_matrix):
        interaction_matrix = interaction_matrix.tocsr()

    model = AlternatingLeastSquares(
        factors=params["factors"],
        regularization=params["regularization"],
        alpha=params["alpha"],
        iterations=10,
        random_state=42
    )

    model.fit(interaction_matrix)

    state["model"] = model
    return state
