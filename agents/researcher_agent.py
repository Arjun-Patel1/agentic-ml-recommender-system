class ResearcherAgent:
    """
    Placeholder for research steps.
    Currently deterministic: returns known best models and metrics.
    """
    def research(self, problem_summary=None):
        insights = {
            "suggested_models": ["ALS", "BPR"],
            "metric_tips": {"recall": 0.01, "precision": 0.001},
            "notes": "MovieLens task, top-K recommendation."
        }
        return insights
