# agents/decision_agent.py

from typing import TypedDict, Dict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
import mlflow
from models.als import train_als
from models.bpr import train_bpr

# -----------------------------
# STATE DEFINITION
# -----------------------------
class AgentState(TypedDict):
    iteration: int
    max_iterations: int
    model_name: str
    metrics: Dict[str, float]
    decision: str
    reasoning: str

# -----------------------------
# LLM (USED ONLY FOR EXPLANATION)
# -----------------------------
llm = OllamaLLM(
    model="mistral",
    temperature=0.2
)

# -----------------------------
# PLANNER
# -----------------------------
def planner(state: AgentState) -> AgentState:
    """
    Decides which model to try next based on metrics.
    """
    metrics = state.get("metrics", {})
    recall = metrics.get("recall_20", 0.0)

    # Simple logic: switch model if recall < 0.1
    if recall < 0.1:
        # Cycle through models
        next_model = {
            "ALS": "NeuralCF",
            "NeuralCF": "BPR",
            "BPR": "ALS"
        }
        state["model_name"] = next_model.get(state.get("model_name", "ALS"), "ALS")

    return state

# -----------------------------
# EXECUTOR
# -----------------------------
from mlflow.tracking import MlflowClient

def executor(state):
    client = MlflowClient()

    # Get default experiment
    experiment = client.get_experiment_by_name("Default")
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.recall_20 DESC"],
        max_results=1
    )

    if not runs:
        return {
            "decision": "RETRY",
            "metrics": {"recall_20": 0.0, "precision_20": 0.0},
            "reasoning": "No MLflow runs found. Retrying training."
        }

    best_run = runs[0]
    metrics = best_run.data.metrics
    params = best_run.data.params

    recall = metrics.get("recall_20", 0.0)
    precision = metrics.get("precision_20", 0.0)
    model_name = params.get("model_name", "unknown")

    decision = "ACCEPT" if recall >= 0.08 else "RETRY"

    return {
        "decision": decision,
        "model_name": model_name,
        "metrics": {
            "recall_20": recall,
            "precision_20": precision
        },
        "reasoning": f"Best model is {model_name} with Recall@20={recall}, Precision@20={precision}"
    }

# -----------------------------
# CRITIC (DECISION)
# -----------------------------
def critic(state: AgentState) -> AgentState:
    """
    Makes final decision based on metrics and iteration.
    """
    metrics = state.get("metrics", {})
    recall = metrics.get("recall_20", 0.0)

    # HARD RULE
    if recall >= 0.10:
        state["decision"] = "ACCEPT"
    else:
        state["decision"] = "RETRY"

    # LLM-based explanation
    explanation_prompt = f"""
Explain this ML decision concisely.

Model Used: {state['model_name']}
Recall@20: {recall}
Precision@20: {metrics.get('precision_20', 0.0)}
Decision: {state['decision']}
Iteration: {state['iteration']}
"""
    state["reasoning"] = llm.invoke(explanation_prompt)
    state["iteration"] += 1

    return state

# -----------------------------
# ROUTER
# -----------------------------
def router(state: AgentState):
    """
    Decide whether to continue or stop.
    """
    if state["decision"] == "ACCEPT":
        return END
    if state["iteration"] >= state["max_iterations"]:
        return END
    return "planner"

# -----------------------------
# BUILD AGENT
# -----------------------------
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("critic", critic)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "critic")
    graph.add_conditional_edges("critic", router)

    return graph.compile()
import mlflow

from utils.config import (
    DEPLOY_RECALL_THRESHOLD,
    STAGING_DIR,
    PRODUCTION_DIR,
)
from utils.model_io import save_model


def autonomous_decision(results: dict):
    """
    results = {
        "ALS": {
            "model": als_model,
            "recall": 0.061,
            "precision": 0.054
        },
        "BPR": {
            "model": bpr_model,
            "recall": 0.0838,
            "precision": 0.136
        }
    }
    """

    best_model_name = max(results, key=lambda x: results[x]["recall"])
    best_result = results[best_model_name]

    print(f"Best Model: {best_model_name}")
    print(f"Recall@20: {best_result['recall']}")
    print(f"Precision@20: {best_result['precision']}")

    # Always save to staging
    save_model(
        model=best_result["model"],
        metrics={
            "recall": best_result["recall"],
            "precision": best_result["precision"],
        },
        model_name=best_model_name,
        directory=STAGING_DIR,
    )

    if best_result["recall"] >= DEPLOY_RECALL_THRESHOLD:
        save_model(
            model=best_result["model"],
            metrics={
                "recall": best_result["recall"],
                "precision": best_result["precision"],
            },
            model_name=best_model_name,
            directory=PRODUCTION_DIR,
        )
        decision = "DEPLOY"
    else:
        decision = "REJECT"

    print(f"Decision: {decision}")
    return decision
