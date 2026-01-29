import numpy as np

class RerankAgent:
    """
    Simple re-ranker:
    boosts confident ALS scores and penalizes low-activity items
    """

    def act(self, state):
        model = state["model"]
        interaction = state["interaction_matrix"]

        user_id = 0
        scores = model.recommend(
            user_id,
            interaction[user_id],
            N=20,
            filter_already_liked_items=True
        )

        reranked = []
        for item, score in scores:
            popularity = interaction[:, item].sum()
            final_score = score * (1 + np.log1p(popularity))
            reranked.append((item, final_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        state["reranked_items"] = reranked[:10]

        return state
