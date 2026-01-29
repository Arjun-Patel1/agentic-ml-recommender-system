import faiss
import numpy as np

class RetrievalAgent:
    def act(self, state: dict) -> dict:
        """
        Builds FAISS index over item embeddings.
        """

        if "item_factors" not in state:
            raise RuntimeError(
                "RetrievalAgent requires item_factors in state"
            )

        print("[RetrievalAgent] Building FAISS index...")

        item_embeddings = state["item_factors"].astype("float32")
        dim = item_embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(item_embeddings)
        index.add(item_embeddings)

        state["faiss_index"] = index
        return state
