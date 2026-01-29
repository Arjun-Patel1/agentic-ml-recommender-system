# agents/registry_agent.py

import mlflow
from mlflow.tracking import MlflowClient


def act(state: dict) -> dict:
    print("[RegistryAgent] Deciding model promotion...")

    recall = state.get("recall@10", 0.0)
    model = state.get("model")

    # Guard: if model missing, skip safely
    if model is None:
        print("[RegistryAgent] No model found, skipping registry step.")
        state["registry_status"] = "skipped"
        return state

    # Promotion threshold (you can tune this)
    PROMOTION_THRESHOLD = 0.20

    if recall < PROMOTION_THRESHOLD:
        print("[RegistryAgent] Model did not meet promotion threshold.")
        state["registry_status"] = "rejected"
        return state

    # Log model
    with mlflow.start_run(nested=True):
        mlflow.log_metric("recall_10", recall)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="AgenticRecommender"
        )

    # Promote model
    client = MlflowClient()
    latest_version = client.get_latest_versions(
        "AgenticRecommender", stages=["None"]
    )[0].version

    client.transition_model_version_stage(
        name="AgenticRecommender",
        version=latest_version,
        stage="Production"
    )

    print("[RegistryAgent] Model promoted to Production ✅")

    state["registry_status"] = "promoted"
    return state
